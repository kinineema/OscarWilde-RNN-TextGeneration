
# Final Model Comparison

## Quantitative Model Comparison

| Model     |   Validation Loss |   Validation Accuracy |   Perplexity |   Training Time (s) |   Epochs |
|:----------|------------------:|----------------------:|-------------:|--------------------:|---------:|
| SimpleRNN |            5.7324 |                0.1227 |       308.7  |               63.15 |        7 |
| BiRNN     |            5.9382 |                0.0971 |       379.23 |               71.06 |        5 |
| LSTM      |            5.7811 |                0.1248 |       324.1  |              236.31 |       10 |
| GRU       |            5.683  |                0.1338 |       293.82 |              149.57 |        8 |

## Temperature-Based Qualitative Comparison


| Model | T=0.5 | T=1.0 | T=1.5 |
|---|---|---|---|
| BiRNN | low diversity, repetitive | much better diversity, very low repetition | highest diversity, but incoherent |
| GRU | balanced, but repetitive | **strong balance of diversity + coherence** | diverse but unstable |
| LSTM | moderate coherence and repetition | highest diversity at 1.0 | high diversity, weak coherence |
| SimpleRNN | moderate diversity | good diversity | very high diversity, poor coherence |


## Key Findings

- **Best validation loss:** GRU
- **Lowest perplexity:** GRU
- **Highest validation accuracy:** GRU
- **Fastest training model:** SimpleRNN
- **Best overall generation setting:** GRU at temperature 1.0

## Final Interpretation

GRU provided the strongest overall performance across the quantitative evaluation metrics.

It achieved the lowest validation loss and perplexity while also obtaining the highest validation accuracy.

The temperature experiments showed a consistent trade-off:

- Temperature 0.5 produced more coherent but more repetitive text.
- Temperature 1.0 provided the strongest balance between coherence, lexical diversity, contextual consistency, and repetition.
- Temperature 1.5 increased lexical diversity but substantially reduced grammatical and contextual coherence.

Although some models achieved higher lexical diversity at temperature 1.5, the generated text became fragmented and semantically unstable.

Therefore, **GRU with temperature 1.0** was selected as the most effective autoregressive text-generation configuration for this dataset.
