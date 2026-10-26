# ACP + probabilidades y tres redes con fusiones0..40

Unidad confirmada por el usuario:porcentaje de incidencia. Rango operativo solicitado0..40; observaciones originales0..36enteras. No se alteran ni redondean las respuestas.

## Entradas y modelos
Continuas estandarizadas→ACP95%; tres probabilidades base estandarizadas aparte yOOF en train; indicadores/calendario comunes. Los polinomios completos2/3incluyen productosPC×probabilidad; la armónica múltiple incluye senos/cosenos anuales y sus interacciones conPC/probabilidades. GP es regresorMSE de3niveles con escalaafín train, distinto delGPclasificador.

## Tres redes
Mismas entradasACP+p; redes36-18,6-9-18-36y10-5-10-5-10-5,80epochs,Adam. Cada red tiene salida40*sigmoid. Entrenamiento:11900positivos+100negativos; validación yprueba completas. Fusores sobre las tres salidasOOF: GAM de splines con enlace logit fraccional, polinomios2/3con enlace logit, PIG truncada0..40y promedio como control.

## Resultado de fusiones
| fusion              | conjunto   |       R2 |   R2_esperado_uniforme |     MAE |    RMSE |   RMSE_esperado_uniforme |   R2_positivos_reales |   MAE_positivos_reales |   MAE_TP |   n_regresados |   n_FN |
|:--------------------|:-----------|---------:|-----------------------:|--------:|--------:|-------------------------:|----------------------:|-----------------------:|---------:|---------------:|-------:|
| promedio_redes      | validacion | 0.196195 |               0.196195 | 2.81322 | 4.89449 |                  4.89449 |             0.004751  |                5.24865 |  5.23441 |           5764 |     55 |
| promedio_redes      | prueba     | 0.19228  |               0.19228  | 2.67651 | 4.68778 |                  4.68778 |             0.0496813 |                4.93328 |  4.93464 |           8702 |     70 |
| GAM                 | validacion | 0.257971 |               0.257971 | 2.76463 | 4.70266 |                  4.70266 |             0.0396446 |                5.20798 |  5.19325 |           5764 |     55 |
| GAM                 | prueba     | 0.246733 |               0.246733 | 2.64557 | 4.52701 |                  4.52701 |             0.0713296 |                4.92337 |  4.92463 |           8702 |     70 |
| polinomio_2         | validacion | 0.256074 |               0.256074 | 2.75913 | 4.70866 |                  4.70866 |             0.0385929 |                5.19908 |  5.18424 |           5764 |     55 |
| polinomio_2         | prueba     | 0.247477 |               0.247477 | 2.63414 | 4.52477 |                  4.52477 |             0.0732575 |                4.90405 |  4.9051  |           8702 |     70 |
| polinomio_3         | validacion | 0.255225 |               0.255225 | 2.7608  | 4.71135 |                  4.71135 |             0.0407456 |                5.19494 |  5.18006 |           5764 |     55 |
| polinomio_3         | prueba     | 0.246945 |               0.246945 | 2.63299 | 4.52637 |                  4.52637 |             0.0760888 |                4.89357 |  4.89452 |           8702 |     70 |
| Poisson_IG_truncada | validacion | 0.260447 |               0.260447 | 2.75338 | 4.6948  |                  4.6948  |             0.0380182 |                5.20053 |  5.18572 |           5764 |     55 |
| Poisson_IG_truncada | prueba     | 0.252428 |               0.252428 | 2.62458 | 4.50986 |                  4.50986 |             0.0747747 |                4.89187 |  4.8928  |           8702 |     70 |

## Resultado de regresores
| familia   | base               | modelo             | conjunto   |       R2 |   R2_esperado_uniforme |     MAE |    RMSE |   RMSE_esperado_uniforme |   R2_positivos_reales |   MAE_positivos_reales |   MAE_TP |   n_regresados |   n_FN |       R2_TP |
|:----------|:-------------------|:-------------------|:-----------|---------:|-----------------------:|--------:|--------:|-------------------------:|----------------------:|-----------------------:|---------:|---------------:|-------:|------------:|
| RF        | ACP_control        | lineal_multiple    | validacion | 0.259743 |               0.25974  | 2.75358 | 4.69704 |                  4.69705 |            0.0206576  |                5.29131 |  5.27    |           5600 |    141 | nan         |
| RF        | ACP_control        | log1p_lineal       | validacion | 0.254881 |               0.254878 | 2.74811 | 4.71244 |                  4.71244 |            0.00779289 |                5.29213 |  5.27084 |           5600 |    141 | nan         |
| RF        | ACP_control        | polinomio_2        | validacion | 0.257604 |               0.257601 | 2.75103 | 4.70382 |                  4.70383 |            0.0335647  |                5.25646 |  5.23406 |           5600 |    141 | nan         |
| RF        | ACP_control        | polinomio_3        | validacion | 0.245177 |               0.245175 | 2.75147 | 4.74302 |                  4.74303 |            0.0442616  |                5.19404 |  5.16968 |           5600 |    141 | nan         |
| RF        | ACP_control        | armonica_multiple  | validacion | 0.261619 |               0.261616 | 2.75199 | 4.69108 |                  4.69109 |            0.0353683  |                5.26831 |  5.24628 |           5600 |    141 | nan         |
| RF        | ACP_control        | genetica_regresion | validacion | 0.265787 |               0.265784 | 2.71799 | 4.67782 |                  4.67783 |            0.0290821  |                5.24622 |  5.2235  |           5600 |    141 | nan         |
| RF        | ACP_probabilidades | lineal_multiple    | validacion | 0.2597   |               0.259697 | 2.75502 | 4.69717 |                  4.69718 |            0.0213589  |                5.29293 |  5.27167 |           5600 |    141 | nan         |
| RF        | ACP_probabilidades | log1p_lineal       | validacion | 0.254113 |               0.25411  | 2.75037 | 4.71487 |                  4.71487 |            0.008559   |                5.29082 |  5.26949 |           5600 |    141 | nan         |
| RF        | ACP_probabilidades | polinomio_2        | validacion | 0.259473 |               0.25947  | 2.74691 | 4.69789 |                  4.6979  |            0.0368784  |                5.24505 |  5.22229 |           5600 |    141 | nan         |
| RF        | ACP_probabilidades | polinomio_3        | validacion | 0.245623 |               0.245621 | 2.74452 | 4.74162 |                  4.74163 |            0.0495436  |                5.16885 |  5.1437  |           5600 |    141 | nan         |
| RF        | ACP_probabilidades | armonica_multiple  | validacion | 0.262609 |               0.262607 | 2.75124 | 4.68793 |                  4.68794 |            0.0383878  |                5.2578  |  5.23544 |           5600 |    141 | nan         |
| RF        | ACP_probabilidades | genetica_regresion | validacion | 0.265787 |               0.265784 | 2.71799 | 4.67782 |                  4.67783 |            0.0290821  |                5.24622 |  5.2235  |           5600 |    141 | nan         |
| LR        | ACP_control        | lineal_multiple    | validacion | 0.262189 |               0.262186 | 2.77447 | 4.68927 |                  4.68928 |            0.0336932  |                5.26049 |  5.24639 |           5764 |     55 | nan         |
| LR        | ACP_control        | log1p_lineal       | validacion | 0.257209 |               0.257206 | 2.7694  | 4.70507 |                  4.70508 |            0.0209553  |                5.2608  |  5.24671 |           5764 |     55 | nan         |
| LR        | ACP_control        | polinomio_2        | validacion | 0.259819 |               0.259816 | 2.77205 | 4.6968  |                  4.6968  |            0.0466723  |                5.22425 |  5.20971 |           5764 |     55 | nan         |
| LR        | ACP_control        | polinomio_3        | validacion | 0.24621  |               0.246207 | 2.77639 | 4.73978 |                  4.73979 |            0.0576469  |                5.1625  |  5.14722 |           5764 |     55 | nan         |
| LR        | ACP_control        | armonica_multiple  | validacion | 0.264702 |               0.264699 | 2.77052 | 4.68128 |                  4.68129 |            0.0484449  |                5.2363  |  5.22191 |           5764 |     55 | nan         |
| LR        | ACP_control        | genetica_regresion | validacion | 0.269141 |               0.269138 | 2.73441 | 4.66713 |                  4.66713 |            0.0414505  |                5.21526 |  5.20062 |           5764 |     55 | nan         |
| LR        | ACP_probabilidades | lineal_multiple    | validacion | 0.26053  |               0.260528 | 2.77762 | 4.69454 |                  4.69455 |            0.0345271  |                5.25519 |  5.24103 |           5764 |     55 | nan         |
| LR        | ACP_probabilidades | log1p_lineal       | validacion | 0.254882 |               0.25488  | 2.77285 | 4.71243 |                  4.71244 |            0.0215491  |                5.25289 |  5.2387  |           5764 |     55 | nan         |
| LR        | ACP_probabilidades | polinomio_2        | validacion | 0.25945  |               0.259448 | 2.77508 | 4.69796 |                  4.69797 |            0.0491008  |                5.22244 |  5.20788 |           5764 |     55 | nan         |
| LR        | ACP_probabilidades | polinomio_3        | validacion | 0.244484 |               0.244481 | 2.77477 | 4.7452  |                  4.74521 |            0.0617773  |                5.14574 |  5.13027 |           5764 |     55 | nan         |
| LR        | ACP_probabilidades | armonica_multiple  | validacion | 0.2629   |               0.262897 | 2.77645 | 4.68701 |                  4.68702 |            0.0501618  |                5.23089 |  5.21643 |           5764 |     55 | nan         |
| LR        | ACP_probabilidades | genetica_regresion | validacion | 0.269141 |               0.269138 | 2.73441 | 4.66713 |                  4.66713 |            0.0414505  |                5.21526 |  5.20062 |           5764 |     55 | nan         |
| GP        | ACP_control        | lineal_multiple    | validacion | 0.23988  |               0.239877 | 2.81543 | 4.75964 |                  4.75964 |            0.00706935 |                5.2724  |  5.20509 |           5760 |    144 | nan         |
| GP        | ACP_control        | log1p_lineal       | validacion | 0.234938 |               0.234935 | 2.81105 | 4.77508 |                  4.77509 |           -0.00539808 |                5.27338 |  5.2061  |           5760 |    144 | nan         |
| GP        | ACP_control        | polinomio_2        | validacion | 0.236662 |               0.23666  | 2.8156  | 4.7697  |                  4.76971 |            0.0202675  |                5.2365  |  5.16804 |           5760 |    144 | nan         |
| GP        | ACP_control        | polinomio_3        | validacion | 0.223392 |               0.22339  | 2.8222  | 4.81098 |                  4.81099 |            0.0332823  |                5.17342 |  5.10293 |           5760 |    144 | nan         |
| GP        | ACP_control        | armonica_multiple  | validacion | 0.243278 |               0.243275 | 2.80835 | 4.74899 |                  4.74899 |            0.0215383  |                5.24853 |  5.18045 |           5760 |    144 | nan         |
| GP        | ACP_control        | genetica_regresion | validacion | 0.246898 |               0.246895 | 2.77063 | 4.73761 |                  4.73762 |            0.0132241  |                5.22462 |  5.15577 |           5760 |    144 | nan         |
| GP        | ACP_probabilidades | lineal_multiple    | validacion | 0.2391   |               0.239097 | 2.82029 | 4.76208 |                  4.76209 |            0.00864351 |                5.27591 |  5.20871 |           5760 |    144 | nan         |
| GP        | ACP_probabilidades | log1p_lineal       | validacion | 0.234038 |               0.234035 | 2.81758 | 4.77789 |                  4.7779  |           -0.00340881 |                5.27831 |  5.21119 |           5760 |    144 | nan         |
| GP        | ACP_probabilidades | polinomio_2        | validacion | 0.237967 |               0.237964 | 2.81526 | 4.76562 |                  4.76563 |            0.0221963  |                5.23825 |  5.16984 |           5760 |    144 | nan         |
| GP        | ACP_probabilidades | polinomio_3        | validacion | 0.223899 |               0.223896 | 2.81114 | 4.80941 |                  4.80942 |            0.0329953  |                5.15934 |  5.08841 |           5760 |    144 | nan         |
| GP        | ACP_probabilidades | armonica_multiple  | validacion | 0.242763 |               0.24276  | 2.81175 | 4.7506  |                  4.75061 |            0.0220545  |                5.25315 |  5.18522 |           5760 |    144 | nan         |
| GP        | ACP_probabilidades | genetica_regresion | validacion | 0.246898 |               0.246895 | 2.77063 | 4.73761 |                  4.73762 |            0.0132241  |                5.22462 |  5.15577 |           5760 |    144 | nan         |
| MIXTA     | ACP_control        | lineal_multiple    | validacion | 0.259616 |               0.259614 | 2.75966 | 4.69744 |                  4.69745 |            0.0229603  |                5.28387 |  5.25308 |           5651 |    111 | nan         |
| MIXTA     | ACP_control        | log1p_lineal       | validacion | 0.254807 |               0.254804 | 2.75395 | 4.71267 |                  4.71268 |            0.0101398  |                5.28428 |  5.25351 |           5651 |    111 | nan         |
| MIXTA     | ACP_control        | polinomio_2        | validacion | 0.257308 |               0.257305 | 2.75695 | 4.70476 |                  4.70476 |            0.0355809  |                5.24904 |  5.2174  |           5651 |    111 | nan         |
| MIXTA     | ACP_control        | polinomio_3        | validacion | 0.244793 |               0.244791 | 2.75834 | 4.74423 |                  4.74424 |            0.0464271  |                5.1876  |  5.15445 |           5651 |    111 | nan         |
| MIXTA     | ACP_control        | armonica_multiple  | validacion | 0.261819 |               0.261816 | 2.75654 | 4.69045 |                  4.69045 |            0.0376162  |                5.25993 |  5.22856 |           5651 |    111 | nan         |
| MIXTA     | ACP_control        | genetica_regresion | validacion | 0.266012 |               0.26601  | 2.72233 | 4.6771  |                  4.67711 |            0.0313842  |                5.23751 |  5.20559 |           5651 |    111 | nan         |
| MIXTA     | ACP_probabilidades | lineal_multiple    | validacion | 0.257368 |               0.257365 | 2.76802 | 4.70457 |                  4.70457 |            0.024176   |                5.28802 |  5.25734 |           5651 |    111 | nan         |
| MIXTA     | ACP_probabilidades | log1p_lineal       | validacion | 0.251847 |               0.251844 | 2.76243 | 4.72202 |                  4.72203 |            0.0109829  |                5.28541 |  5.25466 |           5651 |    111 | nan         |
| MIXTA     | ACP_probabilidades | polinomio_2        | validacion | 0.258301 |               0.258298 | 2.75513 | 4.70161 |                  4.70162 |            0.0381953  |                5.24251 |  5.21071 |           5651 |    111 | nan         |
| MIXTA     | ACP_probabilidades | polinomio_3        | validacion | 0.244669 |               0.244667 | 2.75012 | 4.74462 |                  4.74463 |            0.0487122  |                5.17141 |  5.13787 |           5651 |    111 | nan         |
| MIXTA     | ACP_probabilidades | armonica_multiple  | validacion | 0.262066 |               0.262064 | 2.75704 | 4.68966 |                  4.68967 |            0.0406014  |                5.24914 |  5.21751 |           5651 |    111 | nan         |
| MIXTA     | ACP_probabilidades | genetica_regresion | validacion | 0.266012 |               0.26601  | 2.72233 | 4.6771  |                  4.67711 |            0.0313842  |                5.23751 |  5.20559 |           5651 |    111 | nan         |
| RF        | ACP_control        | lineal_multiple    | prueba     | 0.251488 |               0.251485 | 2.6277  | 4.5127  |                  4.51271 |            0.0535719  |                5.01799 |  5.01377 |           8412 |    197 |   0.0710467 |
| RF        | ACP_control        | log1p_lineal       | prueba     | 0.244487 |               0.244484 | 2.63058 | 4.53375 |                  4.53376 |            0.0421313  |                5.0234  |  5.01934 |           8412 |    197 |   0.0594419 |
| RF        | ACP_control        | polinomio_2        | prueba     | 0.258477 |               0.258475 | 2.60809 | 4.49158 |                  4.49159 |            0.0774132  |                4.95563 |  4.9496  |           8412 |    197 |   0.0952302 |
| RF        | ACP_control        | polinomio_3        | prueba     | 0.252901 |               0.252898 | 2.6019  | 4.50844 |                  4.50844 |            0.100277   |                4.87591 |  4.86755 |           8412 |    197 |   0.118422  |
| RF        | ACP_control        | armonica_multiple  | prueba     | 0.254065 |               0.254062 | 2.6222  | 4.50492 |                  4.50493 |            0.0699387  |                4.98549 |  4.98033 |           8412 |    197 |   0.0876484 |
| RF        | ACP_control        | genetica_regresion | prueba     | 0.264981 |               0.264978 | 2.58511 | 4.47184 |                  4.47185 |            0.0681181  |                4.97657 |  4.97115 |           8412 |    197 |   0.0858017 |
| RF        | ACP_probabilidades | lineal_multiple    | prueba     | 0.250082 |               0.250079 | 2.63141 | 4.51693 |                  4.51694 |            0.0522781  |                5.02483 |  5.02082 |           8412 |    197 |   0.0697344 |
| RF        | ACP_probabilidades | log1p_lineal       | prueba     | 0.242273 |               0.24227  | 2.63499 | 4.54039 |                  4.5404  |            0.0409731  |                5.02674 |  5.02278 |           8412 |    197 |   0.0582671 |
| RF        | ACP_probabilidades | polinomio_2        | prueba     | 0.260067 |               0.260064 | 2.60347 | 4.48676 |                  4.48677 |            0.080199   |                4.94321 |  4.93682 |           8412 |    197 |   0.098056  |
| RF        | ACP_probabilidades | polinomio_3        | prueba     | 0.251178 |               0.251175 | 2.60037 | 4.51363 |                  4.51364 |            0.101065   |                4.86931 |  4.86076 |           8412 |    197 |   0.119221  |
| RF        | ACP_probabilidades | armonica_multiple  | prueba     | 0.254132 |               0.25413  | 2.62107 | 4.50472 |                  4.50473 |            0.0725941  |                4.97059 |  4.96499 |           8412 |    197 |   0.090342  |
| RF        | ACP_probabilidades | genetica_regresion | prueba     | 0.264981 |               0.264978 | 2.58511 | 4.47184 |                  4.47185 |            0.0681181  |                4.97657 |  4.97115 |           8412 |    197 |   0.0858017 |
| LR        | ACP_control        | lineal_multiple    | prueba     | 0.250948 |               0.250945 | 2.66191 | 4.51432 |                  4.51433 |            0.0659387  |                4.98855 |  4.99047 |           8702 |     70 |   0.071017  |
| LR        | ACP_control        | log1p_lineal       | prueba     | 0.243629 |               0.243626 | 2.66574 | 4.53632 |                  4.53633 |            0.0545505  |                4.99393 |  4.9959  |           8702 |     70 |   0.0595767 |
| LR        | ACP_control        | polinomio_2        | prueba     | 0.257564 |               0.257561 | 2.64333 | 4.49434 |                  4.49435 |            0.0898691  |                4.92642 |  4.92771 |           8702 |     70 |   0.0950569 |
| LR        | ACP_control        | polinomio_3        | prueba     | 0.250271 |               0.250268 | 2.64155 | 4.51636 |                  4.51637 |            0.113131   |                4.84573 |  4.8462  |           8702 |     70 |   0.118425  |
| LR        | ACP_control        | armonica_multiple  | prueba     | 0.254194 |               0.254191 | 2.65393 | 4.50453 |                  4.50454 |            0.0822194  |                4.95492 |  4.9565  |           8702 |     70 |   0.0873722 |
| LR        | ACP_control        | genetica_regresion | prueba     | 0.266114 |               0.266111 | 2.61297 | 4.46839 |                  4.4684  |            0.0797652  |                4.94746 |  4.94896 |           8702 |     70 |   0.0849068 |
| LR        | ACP_probabilidades | lineal_multiple    | prueba     | 0.247413 |               0.24741  | 2.66783 | 4.52496 |                  4.52497 |            0.0645376  |                4.98821 |  4.99013 |           8702 |     70 |   0.0696095 |
| LR        | ACP_probabilidades | log1p_lineal       | prueba     | 0.239164 |               0.239161 | 2.67217 | 4.5497  |                  4.5497  |            0.052661   |                4.99111 |  4.99306 |           8702 |     70 |   0.0576786 |
| LR        | ACP_probabilidades | polinomio_2        | prueba     | 0.256154 |               0.256151 | 2.6482  | 4.49861 |                  4.49862 |            0.091372   |                4.92737 |  4.92866 |           8702 |     70 |   0.0965667 |
| LR        | ACP_probabilidades | polinomio_3        | prueba     | 0.248701 |               0.248698 | 2.64194 | 4.52109 |                  4.5211  |            0.11594    |                4.8396  |  4.84    |           8702 |     70 |   0.121247  |
| LR        | ACP_probabilidades | armonica_multiple  | prueba     | 0.251043 |               0.25104  | 2.66018 | 4.51404 |                  4.51405 |            0.0840138  |                4.94325 |  4.94471 |           8702 |     70 |   0.0891748 |
| LR        | ACP_probabilidades | genetica_regresion | prueba     | 0.266114 |               0.266111 | 2.61297 | 4.46839 |                  4.4684  |            0.0797652  |                4.94746 |  4.94896 |           8702 |     70 |   0.0849068 |
| GP        | ACP_control        | lineal_multiple    | prueba     | 0.220979 |               0.220976 | 2.71022 | 4.60374 |                  4.60375 |            0.0270087  |                5.03243 |  4.93861 |           8676 |    208 |   0.0741888 |
| GP        | ACP_control        | log1p_lineal       | prueba     | 0.213862 |               0.213859 | 2.71446 | 4.62473 |                  4.62474 |            0.0162983  |                5.03717 |  4.9435  |           8676 |    208 |   0.0629455 |
| GP        | ACP_control        | polinomio_2        | prueba     | 0.226892 |               0.226889 | 2.69388 | 4.58624 |                  4.58625 |            0.0503665  |                4.97439 |  4.87878 |           8676 |    208 |   0.0987085 |
| GP        | ACP_control        | polinomio_3        | prueba     | 0.218862 |               0.218859 | 2.6944  | 4.61    |                  4.61001 |            0.0748682  |                4.88886 |  4.79061 |           8676 |    208 |   0.124429  |
| GP        | ACP_control        | armonica_multiple  | prueba     | 0.2254   |               0.225397 | 2.69967 | 4.59066 |                  4.59067 |            0.0439917  |                4.99763 |  4.90274 |           8676 |    208 |   0.0920166 |
| GP        | ACP_control        | genetica_regresion | prueba     | 0.236325 |               0.236322 | 2.65598 | 4.55817 |                  4.55818 |            0.0391424  |                4.98836 |  4.89318 |           8676 |    208 |   0.086926  |
| GP        | ACP_probabilidades | lineal_multiple    | prueba     | 0.219681 |               0.219678 | 2.71555 | 4.60758 |                  4.60759 |            0.0279742  |                5.03696 |  4.94328 |           8676 |    208 |   0.0752023 |
| GP        | ACP_probabilidades | log1p_lineal       | prueba     | 0.212101 |               0.212098 | 2.72226 | 4.6299  |                  4.62991 |            0.0172256  |                5.04507 |  4.95164 |           8676 |    208 |   0.0639189 |
| GP        | ACP_probabilidades | polinomio_2        | prueba     | 0.228448 |               0.228445 | 2.69435 | 4.58162 |                  4.58163 |            0.0531647  |                4.97661 |  4.88107 |           8676 |    208 |   0.101646  |
| GP        | ACP_probabilidades | polinomio_3        | prueba     | 0.22065  |               0.220647 | 2.68205 | 4.60472 |                  4.60473 |            0.0768319  |                4.87158 |  4.77279 |           8676 |    208 |   0.12649   |
| GP        | ACP_probabilidades | armonica_multiple  | prueba     | 0.224635 |               0.224632 | 2.70383 | 4.59293 |                  4.59294 |            0.0445397  |                5.00284 |  4.9081  |           8676 |    208 |   0.0925918 |
| GP        | ACP_probabilidades | genetica_regresion | prueba     | 0.236325 |               0.236322 | 2.65598 | 4.55817 |                  4.55818 |            0.0391424  |                4.98836 |  4.89318 |           8676 |    208 |   0.086926  |
| MIXTA     | ACP_control        | lineal_multiple    | prueba     | 0.249509 |               0.249507 | 2.64256 | 4.51866 |                  4.51866 |            0.054688   |                5.01872 |  5.00074 |           8512 |    148 |   0.0722854 |
| MIXTA     | ACP_control        | log1p_lineal       | prueba     | 0.242424 |               0.242421 | 2.64561 | 4.53994 |                  4.53994 |            0.0431957  |                5.02424 |  5.00638 |           8512 |    148 |   0.0606617 |
| MIXTA     | ACP_control        | polinomio_2        | prueba     | 0.25653  |               0.256527 | 2.6225  | 4.49747 |                  4.49748 |            0.0784257  |                4.95649 |  4.93715 |           8512 |    148 |   0.0962946 |
| MIXTA     | ACP_control        | polinomio_3        | prueba     | 0.250637 |               0.250635 | 2.61704 | 4.51526 |                  4.51527 |            0.101414   |                4.87592 |  4.85484 |           8512 |    148 |   0.119546  |
| MIXTA     | ACP_control        | armonica_multiple  | prueba     | 0.252636 |               0.252633 | 2.63506 | 4.50923 |                  4.50924 |            0.0712128  |                4.9845  |  4.96578 |           8512 |    148 |   0.0889992 |
| MIXTA     | ACP_control        | genetica_regresion | prueba     | 0.263821 |               0.263818 | 2.59702 | 4.47536 |                  4.47537 |            0.0693274  |                4.97574 |  4.95683 |           8512 |    148 |   0.0870922 |
| MIXTA     | ACP_probabilidades | lineal_multiple    | prueba     | 0.245268 |               0.245265 | 2.65337 | 4.53141 |                  4.53142 |            0.0532574  |                5.02822 |  5.01045 |           8512 |    148 |   0.0708385 |
| MIXTA     | ACP_probabilidades | log1p_lineal       | prueba     | 0.23766  |               0.237658 | 2.65608 | 4.55419 |                  4.5542  |            0.0418157  |                5.02949 |  5.01174 |           8512 |    148 |   0.0592659 |
| MIXTA     | ACP_probabilidades | polinomio_2        | prueba     | 0.257473 |               0.25747  | 2.61906 | 4.49462 |                  4.49463 |            0.0809516  |                4.94568 |  4.92611 |           8512 |    148 |   0.0988494 |
| MIXTA     | ACP_probabilidades | polinomio_3        | prueba     | 0.252449 |               0.252446 | 2.60477 | 4.5098  |                  4.50981 |            0.102832   |                4.86189 |  4.8405  |           8512 |    148 |   0.12098   |
| MIXTA     | ACP_probabilidades | armonica_multiple  | prueba     | 0.252265 |               0.252262 | 2.6338  | 4.51035 |                  4.51036 |            0.0745303  |                4.96478 |  4.94563 |           8512 |    148 |   0.0923546 |
| MIXTA     | ACP_probabilidades | genetica_regresion | prueba     | 0.263821 |               0.263818 | 2.59702 | 4.47536 |                  4.47537 |            0.0693274  |                4.97574 |  4.95683 |           8512 |    148 |   0.0870922 |

## Selección de validación
{
  "mejor": {
    "fusion": "Poisson_IG_truncada",
    "alpha": 0.01,
    "conjunto": "validacion",
    "R2": 0.2604466841886741,
    "R2_esperado_uniforme": 0.2604466841886741,
    "MAE": 2.7533768121751616,
    "RMSE": 4.694802999499597,
    "RMSE_esperado_uniforme": 4.694802999499597,
    "R2_positivos_reales": 0.038018225827434526,
    "MAE_positivos_reales": 5.200534070646285,
    "MAE_TP": 5.185715970337923,
    "n_regresados": 5764,
    "n_FN": 55
  },
  "criterio": "MáximoR2global de validación, empateRMSE; no se selecciona con prueba.",
  "por_fusion": {
    "promedio_redes": {
      "fusion": "promedio_redes",
      "conjunto": "validacion",
      "R2": 0.19619539168641253,
      "R2_esperado_uniforme": 0.19619539168641253,
      "MAE": 2.8132160644104944,
      "RMSE": 4.894494839592643,
      "RMSE_esperado_uniforme": 4.894494839592643,
      "R2_positivos_reales": 0.004750998253938099,
      "MAE_positivos_reales": 5.248650122806508,
      "MAE_TP": 5.234408953579204,
      "n_regresados": 5764,
      "n_FN": 55
    },
    "GAM": {
      "fusion": "GAM",
      "alpha": 0.1,
      "knots": 8,
      "conjunto": "validacion",
      "R2": 0.2579707048339581,
      "R2_esperado_uniforme": 0.2579707048339581,
      "MAE": 2.7646271135350533,
      "RMSE": 4.702655389582858,
      "RMSE_esperado_uniforme": 4.702655389582858,
      "R2_positivos_reales": 0.039644640619749816,
      "MAE_positivos_reales": 5.207980496066544,
      "MAE_TP": 5.193251681434685,
      "n_regresados": 5764,
      "n_FN": 55
    },
    "polinomio_2": {
      "fusion": "polinomio_2",
      "alpha": 0.0001,
      "degree": 2,
      "conjunto": "validacion",
      "R2": 0.2560736586919258,
      "R2_esperado_uniforme": 0.2560736586919258,
      "MAE": 2.7591314851060003,
      "RMSE": 4.708662874544329,
      "RMSE_esperado_uniforme": 4.708662874544329,
      "R2_positivos_reales": 0.038592912305764226,
      "MAE_positivos_reales": 5.199077780682467,
      "MAE_TP": 5.184242218863748,
      "n_regresados": 5764,
      "n_FN": 55
    },
    "polinomio_3": {
      "fusion": "polinomio_3",
      "alpha": 0.0001,
      "degree": 3,
      "conjunto": "validacion",
      "R2": 0.2552248519596022,
      "R2_esperado_uniforme": 0.2552248519596022,
      "MAE": 2.760797324534514,
      "RMSE": 4.711348359062445,
      "RMSE_esperado_uniforme": 4.711348359062445,
      "R2_positivos_reales": 0.04074558748384405,
      "MAE_positivos_reales": 5.194940604739236,
      "MAE_TP": 5.18005543649434,
      "n_regresados": 5764,
      "n_FN": 55
    },
    "Poisson_IG_truncada": {
      "fusion": "Poisson_IG_truncada",
      "alpha": 0.01,
      "conjunto": "validacion",
      "R2": 0.2604466841886741,
      "R2_esperado_uniforme": 0.2604466841886741,
      "MAE": 2.7533768121751616,
      "RMSE": 4.694802999499597,
      "RMSE_esperado_uniforme": 4.694802999499597,
      "R2_positivos_reales": 0.038018225827434526,
      "MAE_positivos_reales": 5.200534070646285,
      "MAE_TP": 5.185715970337923,
      "n_regresados": 5764,
      "n_FN": 55
    }
  }
}

## Límites
PIG es modelo exploratorio de puntos porcentuales enteros; no conteo de hojas. La prevalencia de train cambia con100ceros. OOF en capas no equivale a validación anidada completa de toda la historia. La prueba histórica ha sido reutilizada; falta cohorte independiente. Los resultados no prueban mejora por complejidad.