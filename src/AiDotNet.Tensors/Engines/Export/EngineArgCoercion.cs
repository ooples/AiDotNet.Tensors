using System;
using System.Collections;
using System.Collections.Generic;
using System.Linq;
using AiDotNet.Tensors.Helpers;
using AiDotNet.Tensors.LinearAlgebra;

namespace AiDotNet.Tensors.Engines.Export;

/// <summary>
/// Converts a portable argument (as a program stores it) to the exact parameter type the generated invoker passes to
/// an IEngine method. The target type always comes from the generated call site, never from the saved program, so a
/// program cannot name a type to instantiate: enums resolve against the declared parameter type, scalars convert to the
/// replay element type, and arrays and lists convert to the declared collection shape.
/// </summary>
public static class EngineArgCoercion
{
    /// <summary>Converts <paramref name="value"/> to <typeparamref name="TParam"/>.</summary>
    public static TParam As<TParam>(object? value)
    {
        if (value is null) return default!;
        if (value is TParam direct) return direct;
        return (TParam)Convert(value, typeof(TParam))!;
    }

    private static object? Convert(object value, Type target)
    {
        if (target.IsInstanceOfType(value)) return value;
        var underlying = Nullable.GetUnderlyingType(target);
        if (underlying is not null) return Convert(value, underlying);

        if (target == typeof(FusedActivationParams) && value is object?[] { Length: 3 } ap)
        {
            float? F(object? o) => o is null ? null : System.Convert.ToSingle(o, System.Globalization.CultureInfo.InvariantCulture);
            return new FusedActivationParams { Alpha = F(ap[0]), Beta = F(ap[1]), Theta = F(ap[2]) };
        }
        if (value is EnumLiteral literal) value = literal.Member;
        if (target.IsEnum) return ToEnum(value, target);

        if (value is double or float or int or long or Half && IsNumeric(target))
            return ConvertScalar(value, target);

        if (target.IsArray && value is IEnumerable seq && value is not string)
        {
            var elem = target.GetElementType()!;
            var items = seq.Cast<object?>().ToList();
            var arr = Array.CreateInstance(elem, items.Count);
            for (int i = 0; i < items.Count; i++) arr.SetValue(items[i] is null ? null : Convert(items[i]!, elem), i);
            return arr;
        }

        if (target.IsGenericType && value is IEnumerable list && value is not string)
        {
            var def = target.GetGenericTypeDefinition();
            if (def == typeof(List<>) || def == typeof(IList<>) || def == typeof(IReadOnlyList<>)
                || def == typeof(IEnumerable<>) || def == typeof(ICollection<>) || def == typeof(IReadOnlyCollection<>))
            {
                var elem = target.GetGenericArguments()[0];
                var result = (IList)Activator.CreateInstance(typeof(List<>).MakeGenericType(elem))!;
                foreach (var item in list) result.Add(item is null ? null : Convert(item, elem));
                return result;
            }
        }

        throw new InvalidCastException($"A program argument of type {value.GetType().Name} cannot bind to a parameter of type {target.Name}.");
    }

    /// <summary>
    /// Binds a stored enum to <paramref name="target"/>. A program records an enum by member name, so only a defined
    /// member is accepted: a numeric string ("999") or a number is rejected unless it is a defined value, except for a
    /// [Flags] enum, whose value may be any combination of defined flags (written as "A, B" or as a number).
    /// </summary>
    private static object ToEnum(object value, Type target)
    {
        bool flags = target.IsDefined(typeof(FlagsAttribute), inherit: false);
        object result;
        if (value is string s)
        {
            var trimmed = s.Trim();
            bool numeric = trimmed.Length > 0 && (char.IsDigit(trimmed[0]) || trimmed[0] is '-' or '+');
            if (numeric && !flags && !long.TryParse(trimmed, System.Globalization.NumberStyles.Integer, System.Globalization.CultureInfo.InvariantCulture, out _))
                throw new InvalidCastException($"'{s}' is not a member of {target.Name}.");
            try { result = Enum.Parse(target, trimmed, ignoreCase: false); }
            catch (ArgumentException) { throw new InvalidCastException($"'{s}' is not a member of {target.Name}."); }
            catch (OverflowException) { throw new InvalidCastException($"'{s}' is out of range for {target.Name}."); }
        }
        else
        {
            long raw;
            try { raw = System.Convert.ToInt64(value, System.Globalization.CultureInfo.InvariantCulture); }
            catch (Exception ex) when (ex is OverflowException or FormatException or InvalidCastException)
            {
                throw new InvalidCastException($"A {value.GetType().Name} cannot bind to {target.Name}.");
            }
            result = Enum.ToObject(target, raw);
        }

        if (Enum.IsDefined(target, result)) return result;
        if (flags && IsFlagCombination(target, result)) return result;
        throw new InvalidCastException($"{value} is not a defined member of {target.Name}.");
    }

    // Every set bit of the value belongs to a defined member.
    private static bool IsFlagCombination(Type target, object value)
    {
        ulong bits = ToBits(value);
        ulong known = 0;
        foreach (var member in Enum.GetValues(target)) known |= ToBits(member!);
        return (bits & ~known) == 0;
    }

    private static ulong ToBits(object enumValue) => Type.GetTypeCode(Enum.GetUnderlyingType(enumValue.GetType())) switch
    {
        TypeCode.UInt64 => System.Convert.ToUInt64(enumValue, System.Globalization.CultureInfo.InvariantCulture),
        _ => unchecked((ulong)System.Convert.ToInt64(enumValue, System.Globalization.CultureInfo.InvariantCulture)),
    };

    private static bool IsNumeric(Type t) =>
        t == typeof(double) || t == typeof(float) || t == typeof(int) || t == typeof(long) || t == typeof(Half)
        || t == typeof(short) || t == typeof(byte) || t == typeof(sbyte) || t == typeof(uint) || t == typeof(ulong)
        || t == typeof(ushort) || t == typeof(decimal) || HasNumericOps(t);

    private static bool HasNumericOps(Type t)
    {
        try { return NumericOpsFor(t) is not null; }
        catch { return false; }
    }

    private static object? NumericOpsFor(Type t) =>
        typeof(MathHelper).GetMethod(nameof(MathHelper.GetNumericOperations))!.MakeGenericMethod(t).Invoke(null, null);

    private static object ConvertScalar(object value, Type target)
    {
        double d = value is Half h ? (double)h : System.Convert.ToDouble(value, System.Globalization.CultureInfo.InvariantCulture);
        if (target == typeof(double)) return d;
        if (target == typeof(float)) return (float)d;
        if (target == typeof(Half)) return (Half)d;
        if (target == typeof(int)) return checked((int)d);
        if (target == typeof(long)) return checked((long)d);
        if (target == typeof(decimal)) return (decimal)d;
        if (target.IsPrimitive) return System.Convert.ChangeType(d, target, System.Globalization.CultureInfo.InvariantCulture);
        var ops = NumericOpsFor(target)!;
        return ops.GetType().GetMethod("FromDouble", new[] { typeof(double) })!.Invoke(ops, new object[] { d })!;
    }
}
