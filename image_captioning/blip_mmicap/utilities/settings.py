from os.path import join


ROOT = "/volume"
TRAIN_NAME = "train"
VAL_NAME = "val"
TEST_NAME = "test"

## Annotation Ground Truth
ANNOTATION_VAL_GT = "annotation_val_gt.json"
ANNOTATION_TEST_GT = "annotation_test_gt.json"

## Annotation trainig
ANNOTATION_TRAIN = "elscap_train.json"
ANNOTATION_TEST = "elscap_test.json"
ANNOTATION_VAL = "elscap_val.json"

# PARAGRAPHS
T5MODEL_NAME = "t5-base"
MAX_WORDS_PARAGRAPH = 512
MIN_WORDS_CAPTION = 5
MIN_WORDS_PARRAGRAPH = 5