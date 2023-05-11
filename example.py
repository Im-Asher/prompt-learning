import torch.nn as nn

from openprompt.data_utils import InputExample
from openprompt.plms import load_plm
from openprompt.prompts import ManualTemplate,ManualVerbalizer
from openprompt import PromptForClassification,PromptDataLoader


classes = [
    "negative",
    "positive"
]

dataset = [
    InputExample(
        guid=0, text_a="Albert Einstein was one of the greatest intellects of his time."),
    InputExample(guid=1, text_a="The film was badly made."),
]

plm, tokenizer, model_config, WrapperClass = load_plm(
    "bert", "bert-base-cased")

promptTemplate = ManualTemplate(
    text="{'placeholder':'text_a'} It was {'mask'}",
    tokenizer=tokenizer,
)

promptVerbalizer = ManualVerbalizer(
    classes=classes,
    label_words={
        "negative":["bad"],
        "positive":["good","wonderful","great"],
    },
    tokenizer=tokenizer
)

promptModel = PromptForClassification(
    template=promptTemplate,
    plm=plm,
    verbalizer=promptVerbalizer
)

data_loader = PromptDataLoader(
    dataset=dataset,
    tokenizer=tokenizer,
    template=promptTemplate,
    tokenizer_wrapper_class=WrapperClass
)


if __name__ == "__main__":
    promptModel.train()
    for item in data_loader:
        y= promptModel(item)
        sfm = nn.Softmax(dim=1)
        pre = sfm(y)
        print(pre)
