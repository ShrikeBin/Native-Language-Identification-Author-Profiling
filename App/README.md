# Recommended Models To Train:

Model training scripts were optimized to work on RTX 3060ti with 8gb of VRAM

### Age:
- DistillBERT Full Fine Tune

### Gender:
- DistillBERT Full Fine Tune
- RoBERTa Full Fine Tune (if you have resources (1% difference))

### Language:
- RoBERTa Full Fine tune (best balance, transformer only)
- CNN on RoBERTa (very cheap, 1% decrease)
- CNN + RoBERTa (good fit combines both worlds) (BEST ONE)
- CNN + RoBERTa Large (Highest Accuracy on the DataSet (81%)) (BUT SUCKS)

### MBTI:
- DistillBERT Full Fine Tune

### Political:
- DistillBERT Full Fine Tune
