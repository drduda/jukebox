# Token Generation
This repo provides code for converting the [FMA Dataset](https://github.com/mdeff/fma) to token sequences by using the encoder of [Jukebox by OpenAI](https://openai.com/blog/jukebox/). 

```
python token_dataset_generator.py --target=genre --size=small --audio_dir=PATH_TO_FMA_DIR
```
`target` can only be genre so far. Each record is one of 8 genres.

`size` can be `small`, `medium` or `large` depending on the size of theFMA Dataset that should be converted. 

`audio_dir` is the path to the directory where 'fma_metadata' and the fma_records are stored.
