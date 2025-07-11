from transformers import pipeline, set_seed

set_seed(42)

models = [
    "Vamsi/T5_Paraphrase_Paws",
    "ramsrigouthamg/t5_paraphraser",
    "prithivida/parrot_paraphraser_on_T5"
]

paraphrasers = []
for model in models:
    try:
        paraphraser = pipeline("text2text-generation", model=model)
        paraphrasers.append((model, paraphraser))
    except Exception as e:
        paraphrasers.append((model, str(e)))

text_1 = (
    "Today is our dragon boat festival, in our Chinese culture, to celebrate it with all safe and great in our lives. "
    "Hope you too, to enjoy it as my deepest wishes. Thank your message to show our words to the doctor, as his next contract checking, to all of us. "
    "I got this message to see the approved message. In fact, I have received the message from the professor, to show me, this, a couple of days ago. "
    "I am very appreciated the full support of the professor, for our Springer proceedings publication."
)

text_2 = (
    "During our final discuss, I told him about the new submission — the one we were waiting since last autumn, but the updates was confusing "
    "as it not included the full feedback from reviewer or maybe editor? Anyway, I believe the team, although bit delay and less communication at recent days, "
    "they really tried best for paper and cooperation. We should be grateful, I mean all of us, for the acceptance and efforts until the Springer link came finally last week, I think. "
    "Also, kindly remind me please, if the doctor still plan for the acknowledgments section edit before he sending again. "
    "Because I didn't see that part final yet, or maybe I missed, I apologize if so. "
    "Overall, let us make sure all are safe and celebrate the outcome with strong coffee and future targets."
)

results = {"text_1": {}, "text_2": {}}
for model_name, paraphraser in paraphrasers:
    if isinstance(paraphraser, str):
        results["text_1"][model_name] = paraphraser
        results["text_2"][model_name] = paraphraser
    else:
        results["text_1"][model_name] = paraphraser(
            text_1, max_length=512, do_sample=True, top_k=120, top_p=0.95)[0]["generated_text"]
        results["text_2"][model_name] = paraphraser(
            text_2, max_length=512, do_sample=True, top_k=120, top_p=0.95)[0]["generated_text"]

for key in results:
    print(f"\n--- {key.upper()} ---")
    for model, output in results[key].items():
        print(f"\n[{model}]\n{output}")
