 # WORKFLOW

 1. save_data.sh: extracts data with specified parameters
 2. main.sh: runs `main.py` and effectively trains N (i.e. 100) models on chosen data (specify type input ('mfcc'), size window, step size and # mels for MFCC/mel spectrogram/spectrogram extraction)
 3. a) lda.sh: gets LDA score on each exp. item from Wu&Holt, LDA is trained on all 'ae' and 'eh' vowels of Timit Test
 3. b) dec.sh: gets dec score on each exp. item from Wu&Holt with a decision model

