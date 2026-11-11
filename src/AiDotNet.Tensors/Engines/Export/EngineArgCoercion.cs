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
        if (target.IsEnum)
        {
            if (value is string s && long.TryParse(s, System.Globalization.NumberStyles.Integer, System.Globalization.CultureInfo.InvariantCulture, out var raw))
                return Enum.ToObject(target, raw);
            return value switch
            {
                string name => Enum.Parse(target, name, ignoreCase: false),
                _ => Enum.ToObject(target, System.Convert.ToInt64(value, System.Globalization.CultureInfo.InvariantCulture)),
            };
        }

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
