# Main results: physical RMSE and ACC

SphereTTC parameters were selected on 2017 and frozen before evaluation on 2018–2019.
RMSE is cosine-latitude weighted and reported in each variable's native physical unit.
ACC is cosine-latitude weighted against the 1979–2016 fit-split daily
month/day climatology; no validation or test targets enter the climatology.
RMSE is not averaged across variables with incompatible units; the complete 49-variable,
5-lead table is in `MAIN_RESULTS.csv`, with year-specific values in
`MAIN_RESULTS_BY_YEAR.csv`.


## z500 at 120h (m^2 s^-2)

| Model | Raw RMSE | SphereTTC RMSE | Raw ACC | SphereTTC ACC |
|---|---:|---:|---:|---:|
| convlstm | 769.29195 | 767.96507 | 0.295219 | 0.299442 |
| transformer | 750.34682 | 750.07351 | 0.340599 | 0.341676 |
| fno | 688.26706 | 687.47877 | 0.501210 | 0.502909 |
| vit | 720.54605 | 720.30319 | 0.424167 | 0.424865 |
| cirt | 722.50549 | 722.64309 | 0.439551 | 0.439514 |
| ClimODE-style adapter | 764.65821 | 764.5771 | 0.313357 | 0.313920 |
| fourcastnetv2 | 425.13483 | 420.33476 | 0.855310 | 0.856255 |
| oneforecast | 321.67472 | 318.48031 | 0.917388 | 0.918347 |
| fuxi | 293.27139 | 293.11769 | 0.929808 | 0.929866 |
| pangu | 307.63494 | 306.79134 | 0.924458 | 0.924608 |
| GraphCast-small 1° | 301.83781 | 301.77566 | 0.926606 | 0.926604 |

## t850 at 120h (K)

| Model | Raw RMSE | SphereTTC RMSE | Raw ACC | SphereTTC ACC |
|---|---:|---:|---:|---:|
| convlstm | 3.3558833 | 3.3126395 | 0.212750 | 0.228126 |
| transformer | 3.2662077 | 3.2477411 | 0.253290 | 0.260869 |
| fno | 3.1194268 | 3.0938418 | 0.345601 | 0.359010 |
| vit | 3.1586361 | 3.1529215 | 0.313578 | 0.317110 |
| cirt | 3.1269346 | 3.1242576 | 0.337764 | 0.339844 |
| ClimODE-style adapter | 3.4113258 | 3.3742933 | 0.238695 | 0.242069 |
| fourcastnetv2 | 2.0862677 | 2.04143 | 0.789970 | 0.794065 |
| oneforecast | 1.7318374 | 1.6901665 | 0.858312 | 0.862194 |
| fuxi | 1.5662304 | 1.5421849 | 0.881879 | 0.884719 |
| pangu | 1.6547642 | 1.6160449 | 0.872010 | 0.875139 |
| GraphCast-small 1° | 1.5663226 | 1.5640111 | 0.882940 | 0.883071 |

## u850 at 120h (m s^-1)

| Model | Raw RMSE | SphereTTC RMSE | Raw ACC | SphereTTC ACC |
|---|---:|---:|---:|---:|
| convlstm | 5.2956059 | 5.2885306 | 0.212758 | 0.214824 |
| transformer | 5.2916149 | 5.2783837 | 0.216496 | 0.220345 |
| fno | 4.9970476 | 4.996528 | 0.354438 | 0.354539 |
| vit | 5.0808703 | 5.0809041 | 0.302866 | 0.302833 |
| cirt | 5.0603533 | 5.0600755 | 0.325888 | 0.326023 |
| ClimODE-style adapter | 5.2753903 | 5.2736701 | 0.225810 | 0.226216 |
| fourcastnetv2 | 3.8158893 | 3.7911071 | 0.723203 | 0.722815 |
| oneforecast | 3.278829 | 3.2630164 | 0.804780 | 0.804425 |
| fuxi | 2.9207892 | 2.9207606 | 0.837683 | 0.837664 |
| pangu | 3.136158 | 3.1273368 | 0.821308 | 0.821138 |
| GraphCast-small 1° | 3.0122853 | 3.0095921 | 0.831931 | 0.831914 |

## v850 at 120h (m s^-1)

| Model | Raw RMSE | SphereTTC RMSE | Raw ACC | SphereTTC ACC |
|---|---:|---:|---:|---:|
| convlstm | 4.9529424 | 4.9470512 | 0.128180 | 0.129553 |
| transformer | 4.9185253 | 4.9172831 | 0.148972 | 0.149223 |
| fno | 4.7329763 | 4.7323451 | 0.306213 | 0.306024 |
| vit | 4.8233816 | 4.823033 | 0.214426 | 0.214585 |
| cirt | 4.7795042 | 4.7796477 | 0.271542 | 0.271360 |
| ClimODE-style adapter | 4.953853 | 4.950395 | 0.131394 | 0.132273 |
| fourcastnetv2 | 3.6034469 | 3.5694704 | 0.709150 | 0.708140 |
| oneforecast | 3.1054826 | 3.0737683 | 0.795256 | 0.794785 |
| fuxi | 2.7879518 | 2.7854065 | 0.826808 | 0.827012 |
| pangu | 2.9658715 | 2.9476355 | 0.812319 | 0.812190 |
| GraphCast-small 1° | 2.8538735 | 2.8491221 | 0.822908 | 0.822933 |

## t2m at 120h (K)

| Model | Raw RMSE | SphereTTC RMSE | Raw ACC | SphereTTC ACC |
|---|---:|---:|---:|---:|
| convlstm | 3.1447233 | 2.9186008 | 0.214075 | 0.249866 |
| transformer | 3.328576 | 2.9765149 | 0.196164 | 0.247785 |
| fno | 3.3948183 | 2.8658716 | 0.201220 | 0.276711 |
| vit | 2.9402608 | 2.7590866 | 0.230403 | 0.278390 |
| cirt | 2.9396922 | 2.8142982 | 0.208915 | 0.262341 |
| ClimODE-style adapter | 2.7248535 | 2.6517763 | 0.328340 | 0.334462 |
| fourcastnetv2 | 1.7617282 | 1.7071288 | 0.739493 | 0.750514 |
| oneforecast | 1.668192 | 1.6103469 | 0.784831 | 0.792748 |
| fuxi | 1.3913786 | 1.3382434 | 0.841950 | 0.851359 |
| pangu | 1.4212161 | 1.3527438 | 0.838334 | 0.849541 |
| GraphCast-small 1° | 1.3893948 | 1.3859031 | 0.844425 | 0.844675 |

## mslp at 120h (Pa)

| Model | Raw RMSE | SphereTTC RMSE | Raw ACC | SphereTTC ACC |
|---|---:|---:|---:|---:|
| convlstm | 680.45321 | 679.70957 | 0.228612 | 0.230901 |
| transformer | 669.35244 | 669.28661 | 0.265781 | 0.266236 |
| fno | 622.5423 | 622.69251 | 0.425621 | 0.425294 |
| vit | 644.07974 | 644.04139 | 0.353587 | 0.353792 |
| cirt | 639.06863 | 639.88657 | 0.390564 | 0.388376 |
| ClimODE-style adapter | 678.39858 | 678.21641 | 0.247494 | 0.247527 |
| fourcastnetv2 | 406.44082 | 401.04848 | 0.821806 | 0.822360 |
| oneforecast | 316.80425 | 314.69173 | 0.893118 | 0.893370 |
| fuxi | 289.43116 | 289.02446 | 0.907264 | 0.907483 |
| pangu | 303.47241 | 301.66241 | 0.901202 | 0.901569 |
| GraphCast-small 1° | 297.21333 | 296.30536 | 0.904785 | 0.904971 |
