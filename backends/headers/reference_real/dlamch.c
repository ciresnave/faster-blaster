/* dlamch - machine parameters (double precision) */
double dlamch_(const char *cmach)
{
    char cmach_val = *cmach;
    
    // IEEE 754 double precision parameters
    switch (cmach_val) {
        case 'E': return 2.2204460492503131e-16;   // epsilon (relative machine precision)
        case 'S': return 2.2250738585072014e-308;  // safe minimum (smallest positive usable)
        case 'B': return 2.0;                      // radix (base)
        case 'P': return 4.4408920985006262e-16;   // precision (eps*b)
        case 'N': return 53.0;                     // number of mantissa bits
        case 'R': return 1.0;                      // rounding mode (1.0 for round-to-nearest)
        case 'M': return -1022.0;                  // min exponent
        case 'U': return 2.2250738585072014e-308;  // underflow threshold
        case 'L': return 1.7976931348623157e+308;  // overflow threshold
        case 'O': return 1024.0;                   // max exponent
        default: return 0.0;
    }
}
