// Copyright (c) CrowdTrainer LLC. All rights reserved.

using System;
using System.Runtime.InteropServices;
using AiDotNet.Tensors.Engines.DirectGpu.CUDA.Ptx;

namespace AiDotNet.Tensors.Engines.DirectGpu.CUDA;

/// <summary>
/// The SM issue-rate modifiers the NVIDIA resource manager reports for one GPU (#1549): each field is a speed select,
/// 0 = full rate and n = 1/2^n of it (NV2080_CTRL_GR_GET_SM_ISSUE_RATE_MODIFIER_*_REDUCED_SPEED_1_*). FMLA16 is FP16
/// tensor-core math with FP16 accumulation, FMLA32 the same with FP32 accumulation (and TF32). Stock GeForce Ampere/Ada
/// report FMLA32 = 1 (half rate), the RTX 5090 reports 3 (an eighth); datacenter and workstation parts, and GeForce
/// cards unlocked by a speed-select module, report 0.
/// </summary>
internal readonly record struct SmIssueRateModifier(byte Fmla16, byte Fmla32, byte Dp, byte Ffma)
{
    /// <summary>The slowest speed select the RM defines (REDUCED_SPEED_1_64: 1/64 of the full rate).</summary>
    internal const byte MaxSpeedSelect = 6;

    /// <summary>True when both tensor-core speed selects are within the range the RM defines.</summary>
    internal static bool SpeedSelectsAreDefined(byte fmla16, byte fmla32) =>
        fmla16 <= MaxSpeedSelect && fmla32 <= MaxSpeedSelect;

    /// <summary>
    /// As a rate result in units of the full rate (1.0 = full), so it carries the same <see cref="PtxMmaRateProbe.Result.RateClass"/>
    /// a measurement would: FP16-accumulate over FP32-accumulate = 2^(FMLA32 - FMLA16).
    /// </summary>
    internal PtxMmaRateProbe.Result ToRateResult() =>
        new(Math.Pow(2, -Fmla32), Math.Pow(2, -Fmla16), PtxMmaRateProbe.RateSource.DriverQuery);
}

/// <summary>Reads <see cref="SmIssueRateModifier"/> for the GPU at a PCI address; a seam so routing can be tested.</summary>
internal interface ISmIssueRateQuery
{
    /// <summary>True with the modifiers, or false with why the query could not answer (the caller then measures).</summary>
    bool TryQuery(int pciDomain, int pciBus, int pciDevice, out SmIssueRateModifier modifier, out string failure);
}

/// <summary>
/// The driver's read-only <c>NV2080_CTRL_CMD_GR_GET_SM_ISSUE_RATE_MODIFIER</c> (0x20801230) through the Linux RM
/// ioctl interface: a root client on <c>/dev/nvidiactl</c>, the GPU's <c>/dev/nvidiaN</c> registered to it, a device
/// and a subdevice object, then the control call. It reads the speed selects; it writes nothing to the GPU and needs no
/// privilege beyond opening the device files any CUDA process opens. Structure layouts and constants are NVIDIA's,
/// from the MIT-licensed open-gpu-kernel-modules headers (nvos.h, nv-ioctl.h, nv-ioctl-numbers.h, nv_escape.h,
/// cl0080.h, cl2080.h, ctrl0000gpu.h, ctrl2080gr.h). Any failure (no such device file, an older driver without the
/// control, a refused client) reports false and the caller falls back to <see cref="PtxMmaRateProbe"/>. Windows has no
/// equivalent user-mode path, so it always reports false there.
/// </summary>
internal sealed class NvRmIssueRateQuery : ISmIssueRateQuery
{
    internal static readonly NvRmIssueRateQuery Instance = new();

    private const uint NV01_ROOT_CLIENT = 0x41;
    private const uint NV01_DEVICE_0 = 0x80;
    private const uint NV20_SUBDEVICE_0 = 0x2080;
    private const uint NV0000_CTRL_CMD_GPU_GET_ID_INFO_V2 = 0x205;
    internal const uint NV2080_CTRL_CMD_GR_GET_SM_ISSUE_RATE_MODIFIER = 0x20801230;

    private const int NV_IOCTL_BASE = 200;
    private const int NV_ESC_CARD_INFO = NV_IOCTL_BASE + 0;
    private const int NV_ESC_REGISTER_FD = NV_IOCTL_BASE + 1;
    private const int NV_ESC_ATTACH_GPUS_TO_FD = NV_IOCTL_BASE + 12;
    private const int NV_ESC_RM_FREE = 0x29;
    private const int NV_ESC_RM_CONTROL = 0x2A;
    private const int NV_ESC_RM_ALLOC = 0x2B;
    private const int MaxCards = 32;

    private const int O_RDWR = 2;
    private const int O_CLOEXEC = 0x80000; // Linux x86-64 and aarch64

    // NVOS64_PARAMETERS: hRoot, hObjectParent, hObjectNew, hClass, pAllocParms (8-aligned), pRightsRequested,
    // paramsSize, flags, status.
    [StructLayout(LayoutKind.Explicit, Size = 48)]
    private struct Nvos64
    {
        [FieldOffset(0)] public uint HRoot;
        [FieldOffset(4)] public uint HObjectParent;
        [FieldOffset(8)] public uint HObjectNew;
        [FieldOffset(12)] public uint HClass;
        [FieldOffset(16)] public ulong PAllocParms;
        [FieldOffset(24)] public ulong PRightsRequested;
        [FieldOffset(32)] public uint ParamsSize;
        [FieldOffset(36)] public uint Flags;
        [FieldOffset(40)] public uint Status;
    }

    // NVOS54_PARAMETERS: hClient, hObject, cmd, flags, params (8-aligned), paramsSize, status.
    [StructLayout(LayoutKind.Explicit, Size = 32)]
    private struct Nvos54
    {
        [FieldOffset(0)] public uint HClient;
        [FieldOffset(4)] public uint HObject;
        [FieldOffset(8)] public uint Cmd;
        [FieldOffset(12)] public uint Flags;
        [FieldOffset(16)] public ulong Params;
        [FieldOffset(24)] public uint ParamsSize;
        [FieldOffset(28)] public uint Status;
    }

    // NVOS00_PARAMETERS (free): hRoot, hObjectParent, hObjectOld, status.
    [StructLayout(LayoutKind.Explicit, Size = 16)]
    private struct Nvos00
    {
        [FieldOffset(0)] public uint HRoot;
        [FieldOffset(4)] public uint HObjectParent;
        [FieldOffset(8)] public uint HObjectOld;
        [FieldOffset(12)] public uint Status;
    }

    // nv_ioctl_card_info_t: valid (NvBool), nv_pci_info_t {domain u32, bus, slot, function u8, vendor_id, device_id u16},
    // gpu_id, interrupt_line, four 8-aligned u64 apertures, minor_number, dev_name[10].
    [StructLayout(LayoutKind.Explicit, Size = 72)]
    private struct CardInfo
    {
        [FieldOffset(0)] public byte Valid;
        [FieldOffset(4)] public uint Domain;
        [FieldOffset(8)] public byte Bus;
        [FieldOffset(9)] public byte Slot;
        [FieldOffset(10)] public byte Function;
        [FieldOffset(16)] public uint GpuId;
        [FieldOffset(56)] public uint MinorNumber;
    }

    // NV0000_CTRL_GPU_GET_ID_INFO_V2_PARAMS.
    [StructLayout(LayoutKind.Explicit, Size = 32)]
    private struct GpuIdInfoV2
    {
        [FieldOffset(0)] public uint GpuId;
        [FieldOffset(4)] public uint GpuFlags;
        [FieldOffset(8)] public uint DeviceInstance;
        [FieldOffset(12)] public uint SubDeviceInstance;
    }

    // NV0080_ALLOC_PARAMETERS: deviceId, hClientShare, hTargetClient, hTargetDevice, flags, three 8-aligned u64, vaMode.
    [StructLayout(LayoutKind.Explicit, Size = 56)]
    private struct DeviceAlloc
    {
        [FieldOffset(0)] public uint DeviceId;
        [FieldOffset(4)] public uint HClientShare;
        [FieldOffset(48)] public uint VaMode;
    }

    // NV2080_CTRL_GR_GET_SM_ISSUE_RATE_MODIFIER_PARAMS: grRouteInfo {flags u32, route u64 (8-aligned)}, then nine u8.
    [StructLayout(LayoutKind.Explicit, Size = 32)]
    private struct IssueRateParams
    {
        [FieldOffset(0)] public uint RouteFlags;
        [FieldOffset(8)] public ulong Route;
        [FieldOffset(16)] public byte Imla0;
        [FieldOffset(17)] public byte Fmla16;
        [FieldOffset(18)] public byte Dp;
        [FieldOffset(19)] public byte Fmla32;
        [FieldOffset(20)] public byte Ffma;
    }

    [DllImport("libc", EntryPoint = "open", SetLastError = true)]
    private static extern int SysOpen([MarshalAs(UnmanagedType.LPStr)] string path, int flags);

    [DllImport("libc", EntryPoint = "close", SetLastError = true)]
    private static extern int SysClose(int fd);

    [DllImport("libc", EntryPoint = "ioctl", SetLastError = true)]
    private static extern unsafe int SysIoctl(int fd, UIntPtr request, void* arg);

    // _IOWR('F', nr, size): dir READ|WRITE (3) << 30 | size << 16 | type << 8 | nr, the encoding the driver decodes with
    // _IOC_NR / _IOC_SIZE.
    internal static ulong IoctlRequest(int nr, int size) =>
        (3UL << 30) | ((ulong)(uint)size << 16) | ((ulong)'F' << 8) | (uint)nr;

    private static unsafe bool Ioctl(int fd, int nr, void* arg, int size) =>
        SysIoctl(fd, new UIntPtr(IoctlRequest(nr, size)), arg) == 0;

    public bool TryQuery(int pciDomain, int pciBus, int pciDevice, out SmIssueRateModifier modifier, out string failure)
    {
        modifier = default;
        if (!RuntimeInformation.IsOSPlatform(OSPlatform.Linux))
        {
            failure = "the RM issue-rate query is only available through the Linux driver interface";
            return false;
        }
        try
        {
            return TryQueryLinux(pciDomain, pciBus, pciDevice, out modifier, out failure);
        }
        catch (Exception ex) when (ex is not OutOfMemoryException)
        {
            // Best effort: the caller measures instead. Any P/Invoke failure (a missing or non-ELF libc, a marshaling
            // error) must report false here, not escape into the GEMM call that asked for the rate.
            failure = "the driver query failed (" + ex.GetType().Name + "): " + ex.Message;
            return false;
        }
    }

    private static unsafe bool TryQueryLinux(int pciDomain, int pciBus, int pciDevice,
        out SmIssueRateModifier modifier, out string failure)
    {
        modifier = default;
        int ctl = SysOpen("/dev/nvidiactl", O_RDWR | O_CLOEXEC);
        if (ctl < 0)
        {
            failure = "cannot open /dev/nvidiactl (errno " + Marshal.GetLastWin32Error() + ")";
            return false;
        }
        int dev = -1;
        uint client = 0;
        try
        {
            var cards = new CardInfo[MaxCards];
            fixed (CardInfo* pCards = cards)
            {
                if (!Ioctl(ctl, NV_ESC_CARD_INFO, pCards, sizeof(CardInfo) * MaxCards))
                {
                    failure = "NV_ESC_CARD_INFO failed (errno " + Marshal.GetLastWin32Error() + ")";
                    return false;
                }
            }
            int match = -1;
            for (int i = 0; i < MaxCards; i++)
                if (cards[i].Valid != 0 && cards[i].Domain == (uint)pciDomain && cards[i].Bus == pciBus
                    && cards[i].Slot == pciDevice)
                {
                    match = i;
                    break;
                }
            if (match < 0)
            {
                failure = $"no driver card at PCI {pciDomain:x4}:{pciBus:x2}:{pciDevice:x2}";
                return false;
            }
            var card = cards[match];

            var root = new Nvos64 { HClass = NV01_ROOT_CLIENT };
            if (!Ioctl(ctl, NV_ESC_RM_ALLOC, &root, sizeof(Nvos64)) || root.Status != 0)
            {
                failure = "root client allocation refused (status 0x" + root.Status.ToString("x") + ")";
                return false;
            }
            client = root.HObjectNew;

            dev = SysOpen("/dev/nvidia" + card.MinorNumber, O_RDWR | O_CLOEXEC);
            if (dev < 0)
            {
                failure = "cannot open /dev/nvidia" + card.MinorNumber + " (errno " + Marshal.GetLastWin32Error() + ")";
                return false;
            }
            int ctlFd = ctl;
            if (!Ioctl(dev, NV_ESC_REGISTER_FD, &ctlFd, sizeof(int)))
            {
                failure = "NV_ESC_REGISTER_FD failed (errno " + Marshal.GetLastWin32Error() + ")";
                return false;
            }
            uint gpuId = card.GpuId;
            // Attaching an already attached GPU is harmless; its result is not needed.
            Ioctl(ctl, NV_ESC_ATTACH_GPUS_TO_FD, &gpuId, sizeof(uint));

            var idInfo = new GpuIdInfoV2 { GpuId = gpuId };
            if (!Control(ctl, client, client, NV0000_CTRL_CMD_GPU_GET_ID_INFO_V2, &idInfo, sizeof(GpuIdInfoV2),
                    out uint idStatus))
            {
                failure = "GPU_GET_ID_INFO_V2 failed (status 0x" + idStatus.ToString("x") + ")";
                return false;
            }

            var deviceParams = new DeviceAlloc { DeviceId = idInfo.DeviceInstance, HClientShare = client };
            if (!Alloc(ctl, client, client, NV01_DEVICE_0, &deviceParams, sizeof(DeviceAlloc), out uint device,
                    out uint devStatus))
            {
                failure = "device allocation refused (status 0x" + devStatus.ToString("x") + ")";
                return false;
            }
            uint subDeviceId = idInfo.SubDeviceInstance;
            if (!Alloc(ctl, client, device, NV20_SUBDEVICE_0, &subDeviceId, sizeof(uint), out uint subdevice,
                    out uint subStatus))
            {
                failure = "subdevice allocation refused (status 0x" + subStatus.ToString("x") + ")";
                return false;
            }

            var rates = new IssueRateParams(); // route flags 0: the default GR engine
            if (!Control(ctl, client, subdevice, NV2080_CTRL_CMD_GR_GET_SM_ISSUE_RATE_MODIFIER, &rates,
                    sizeof(IssueRateParams), out uint rateStatus))
            {
                failure = "GET_SM_ISSUE_RATE_MODIFIER failed (status 0x" + rateStatus.ToString("x") + ")";
                return false;
            }
            // A control that "succeeds" but leaves garbage would otherwise be cached as the device's rate and the probe
            // would never run; decline anything outside the documented speed-select range so the caller measures.
            if (!SmIssueRateModifier.SpeedSelectsAreDefined(rates.Fmla16, rates.Fmla32))
            {
                failure = "GET_SM_ISSUE_RATE_MODIFIER returned out-of-range speed selects (fmla16 " + rates.Fmla16
                    + ", fmla32 " + rates.Fmla32 + "; at most " + SmIssueRateModifier.MaxSpeedSelect + " is defined)";
                return false;
            }
            modifier = new SmIssueRateModifier(rates.Fmla16, rates.Fmla32, rates.Dp, rates.Ffma);
            failure = string.Empty;
            return true;
        }
        finally
        {
            if (client != 0)
            {
                // Freeing the client frees its device and subdevice; closing the control fd would free it anyway.
                var free = new Nvos00 { HRoot = client, HObjectParent = client, HObjectOld = client };
                Ioctl(ctl, NV_ESC_RM_FREE, &free, sizeof(Nvos00));
            }
            if (dev >= 0) SysClose(dev);
            SysClose(ctl);
        }
    }

    private static unsafe bool Alloc(int ctl, uint client, uint parent, uint hClass, void* parameters, int size,
        out uint handle, out uint status)
    {
        var p = new Nvos64
        {
            HRoot = client,
            HObjectParent = parent,
            HClass = hClass,
            PAllocParms = (ulong)parameters,
            ParamsSize = (uint)size,
        };
        bool ok = Ioctl(ctl, NV_ESC_RM_ALLOC, &p, sizeof(Nvos64)) && p.Status == 0;
        handle = p.HObjectNew;
        status = p.Status;
        return ok;
    }

    private static unsafe bool Control(int ctl, uint client, uint obj, uint cmd, void* parameters, int size,
        out uint status)
    {
        var p = new Nvos54
        {
            HClient = client,
            HObject = obj,
            Cmd = cmd,
            Params = (ulong)parameters,
            ParamsSize = (uint)size,
        };
        bool ok = Ioctl(ctl, NV_ESC_RM_CONTROL, &p, sizeof(Nvos54)) && p.Status == 0;
        status = p.Status;
        return ok;
    }
}
