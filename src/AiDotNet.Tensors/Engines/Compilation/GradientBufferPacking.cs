using System;
using System.Collections.Generic;

namespace AiDotNet.Tensors.Engines.Compilation;

/// <summary>Assigns whole banks to nonoverlapping inclusive backward-action intervals, largest buffers first.</summary>
internal static class GradientBufferPacking
{
    internal readonly struct Interval
    {
        internal readonly int Id, Elements, First, Last;
        internal Interval(int id, int elements, int first, int last)
        { Id = id; Elements = elements; First = first; Last = last; }
    }

    internal static (int[] BankOf, List<int> Capacities) Pack(int gradientCount,
        List<Interval> intervals, bool longestLifetimeFirst)
    {
        var ordered = new List<Interval>(intervals);
        ordered.Sort((a, b) =>
        {
            int compare = b.Elements.CompareTo(a.Elements);
            if (compare == 0 && longestLifetimeFirst)
                compare = ((long)b.Last - b.First).CompareTo((long)a.Last - a.First);
            if (compare == 0) compare = a.First.CompareTo(b.First);
            if (compare == 0) compare = a.Last.CompareTo(b.Last);
            return compare == 0 ? a.Id.CompareTo(b.Id) : compare;
        });
        var capacities = new List<int>();
        var banks = new List<List<Interval>>();
        var assignment = new int[gradientCount];
        foreach (var interval in ordered)
        {
            if (interval.Id < 0 || interval.Id >= gradientCount || interval.Elements <= 0
                || interval.First < 0 || interval.Last < interval.First)
                throw new ArgumentException("Invalid gradient lifetime.", nameof(intervals));
            int pick = -1;
            for (int bank = 0; bank < banks.Count; bank++)
            {
                bool overlaps = false;
                foreach (var other in banks[bank])
                    if (interval.First <= other.Last && other.First <= interval.Last)
                    { overlaps = true; break; }
                if (overlaps) continue;
                // Sizes descend, so every existing bank already fits. Prefer tight capacity, then a used bank.
                if (pick < 0 || capacities[bank] < capacities[pick]
                    || capacities[bank] == capacities[pick] && banks[bank].Count > banks[pick].Count)
                    pick = bank;
            }
            if (pick < 0)
            {
                pick = banks.Count;
                banks.Add(new List<Interval>());
                capacities.Add(interval.Elements);
            }
            banks[pick].Add(interval);
            assignment[interval.Id] = pick;
        }
        return (assignment, capacities);
    }
}
