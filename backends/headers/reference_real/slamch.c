/* slamch - machine parameters (single precision) */
float slamch_(const char *cmach)
{
    char cmach_val = *cmach;
    
    // IEEE 754 single precision parameters
    switch (cmach_val) {
        case 'E': return 2.2204460e-16f;      // epsilon (relative machine precision)
        case 'S': return 1.1754944e-38f;      // safe minimum (smallest positive usable number)
        case 'B': return 2.0f;                // radix (base)
        case 'P': return 2.4e-7f;             // precision (eps*b)
        case 'N': return 24.0f;               // number of mantissa bits
        case 'R': return 1.0f;                // rounding mode (1.0 for round-to-nearest)
        case 'M': return -126.0f;             // min exponent
        case 'U': return 1.1754944e-38f;      // underflow threshold
        case 'L': return 3.4028235e+38f;      // overflow threshold
        case 'O': return 128.0f;              // max exponent
        default: return 0.0f;
    }
}
