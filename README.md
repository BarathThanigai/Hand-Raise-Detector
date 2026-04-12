# Hand Raise Detector

A standalone browser app (very basic though) that:

- accepts an uploaded image
- detects multiple people
- classifies each person as `hand raised` or `hand not raised`
- shows counts and percentages

## Notes

- The model is loaded from MediaPipe/CDN, so internet is needed for loading.
- Accuracy is best when people are visible from shoulders to wrists.
