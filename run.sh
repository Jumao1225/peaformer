#!/bin/bash

# # w/o surface
# FBDB15K
script -q -c "bash run_meaformer.sh 0 FBDB15K norm 0.8 0" logs/run_FBDB15K08.log
script -q -c "bash run_meaformer.sh 0 FBDB15K norm 0.5 0" logs/run_FBDB15K05.log
script -q -c "bash run_meaformer.sh 0 FBDB15K norm 0.2 0" logs/run_FBDB15K02.log
# FBYG15K
script -q -c "bash run_meaformer.sh 0 FBYG15K norm 0.8 0" logs/run_FBYG15K08.log
script -q -c "bash run_meaformer.sh 0 FBYG15K norm 0.5 0" logs/run_FBYG15K05.log
script -q -c "bash run_meaformer.sh 0 FBYG15K norm 0.2 0" logs/run_FBYG15K02.log
# DBP15K
script -q -c "bash run_meaformer.sh 0 DBP15K zh_en 0.3 0" logs/run_DBP15Kzh_en03.log
script -q -c "bash run_meaformer.sh 0 DBP15K ja_en 0.3 0" logs/run_DBP15Kja_en03.log
script -q -c "bash run_meaformer.sh 0 DBP15K fr_en 0.3 0" logs/run_DBP15Kfr_en03.log
# # w/ surface
# DBP15K
script -q -c "bash run_meaformer.sh 0 DBP15K zh_en 0.3 1" logs/run_DBP15Kzh_en031.log
script -q -c "bash run_meaformer.sh 0 DBP15K ja_en 0.3 1" logs/run_DBP15Kja_en031.log
script -q -c "bash run_meaformer.sh 0 DBP15K fr_en 0.3 1" logs/run_DBP15Kfr_en031.log


# # w/o surface
# FBDB15K
script -q -c "bash run_meaformer_il.sh 0 FBDB15K norm 0.8 0" logs/run_il_FBDB15K08_$(date +%Y%m%d_%H%M%S).log
script -q -c "bash run_meaformer_il.sh 0 FBDB15K norm 0.5 0" logs/run_il_FBDB15K05_$(date +%Y%m%d_%H%M%S).log
script -q -c "bash run_meaformer_il.sh 0 FBDB15K norm 0.2 0" logs/run_il_FBDB15K02_$(date +%Y%m%d_%H%M%S).log
# FBYG15K
script -q -c "bash run_meaformer_il.sh 0 FBYG15K norm 0.8 0" logs/run_il_FBYG15K08_$(date +%Y%m%d_%H%M%S).log
script -q -c "bash run_meaformer_il.sh 0 FBYG15K norm 0.5 0" logs/run_il_FBYG15K05_$(date +%Y%m%d_%H%M%S).log
script -q -c "bash run_meaformer_il.sh 0 FBYG15K norm 0.2 0" logs/run_il_FBYG15K02_$(date +%Y%m%d_%H%M%S).log
# DBP15K
script -q -c "bash run_meaformer_il.sh 0 DBP15K zh_en 0.3 0" logs/run_il_DBP15Kzh_en03_$(date +%Y%m%d_%H%M%S).log
script -q -c "bash run_meaformer_il.sh 0 DBP15K ja_en 0.3 0" logs/run_il_DBP15Kja_en03_$(date +%Y%m%d_%H%M%S).log
script -q -c "bash run_meaformer_il.sh 0 DBP15K fr_en 0.3 0" logs/run_il_DBP15Kfr_en03_$(date +%Y%m%d_%H%M%S).log
# # w/ surface
# DBP15K
script -q -c "bash run_meaformer_il.sh 0 DBP15K zh_en 0.3 1" logs/run_il_DBP15Kzh_en031_$(date +%Y%m%d_%H%M%S).log
script -q -c "bash run_meaformer_il.sh 0 DBP15K ja_en 0.3 1" logs/run_il_DBP15Kja_en031_$(date +%Y%m%d_%H%M%S).log
script -q -c "bash run_meaformer_il.sh 0 DBP15K fr_en 0.3 1" logs/run_il_DBP15Kfr_en031_$(date +%Y%m%d_%H%M%S).log