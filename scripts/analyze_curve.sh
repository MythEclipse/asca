#!/usr/bin/env awk -f
# Hitung mean +- std dan delta berpasangan dari log `asca bench --eval-curve`.
#
# Pemakaian:
#   scripts/analyze_curve.sh /tmp/curve_4m.log
#
# Format masukan (dari cmd_bench):
#     kurva (token, asca, dense):
#          128000  1.8448  2.0776  delta -0.2328
#
# Keluaran: per titik token, rata-rata lintas seed, std, delta berpasangan, dan t.
# df = n-1. Nilai kritis dua sisi: 2,776 (p=0,05) dan 4,604 (p=0,01) untuk n=5.

/^[ \t]+[0-9]+[ \t]+[0-9.-]+[ \t]+[0-9.-]+[ \t]+delta/ {
    tok = $1; a = $2; d = $3; dl = $5;
    na[tok]++; sa[tok] += a; sa2[tok] += a * a;
    nd[tok]++; sden[tok] += d; sden2[tok] += d * d;
    ndl[tok]++; sdel[tok] += dl; sdel2[tok] += dl * dl;
    if (!(tok in seen)) { seen[tok] = 1; order[++n] = tok; }
}

function sd_of(s, s2, k,    m, v) {
    m = s / k;
    v = (s2 - k * m * m) / (k - 1);
    return (v > 0) ? sqrt(v) : 0;
}

END {
    printf "%-9s %5s %19s %19s %19s %7s\n", "token", "n", "ASCA", "dense", "delta", "t";
    for (i = 1; i <= n; i++) {
        k = order[i];
        if (na[k] < 2) continue;
        ma = sa[k] / na[k]; md = sden[k] / nd[k]; mdel = sdel[k] / ndl[k];
        sa_ = sd_of(sa[k], sa2[k], na[k]);
        sd_ = sd_of(sden[k], sden2[k], nd[k]);
        sdl = sd_of(sdel[k], sdel2[k], ndl[k]);
        t = (sdl > 0) ? mdel / (sdl / sqrt(ndl[k])) : 0;
        printf "%-9d %5d %8.4f +- %-6.4f %8.4f +- %-6.4f %8.4f +- %-6.4f %7.2f\n",
               k, na[k], ma, sa_, md, sd_, mdel, sdl, t;
    }
    printf "\ncatatan: df = n-1; n=5 -> kritis dua sisi 2,776 (p=0,05), 4,604 (p=0,01)\n";
}
