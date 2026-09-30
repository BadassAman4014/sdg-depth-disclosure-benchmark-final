# Paper: mnsc.2023.03253.sm1.pdf

Total Pages: 19

## Page 1

 
Online Appendix to: 
 
ChatGPT for Textual Analysis? 
How to use Generative LLMs in Accounting Research 
 
Ties de Kok 
 
June 2024 
 
The following pages provide further details on the items mentioned in the main paper. 
 
OA 1 – GPT model overview and discussion 
ii 
OA 2 – Full prompts 
v 
OA 3 – Full description of the GPT method 
ix 
OA 4 – Examples of classification differences between the GPT and Gow et al. methods 
xii 
OA 5 - Using and fine-tuning a local GLLM 
xiv 
OA 6 – Additional analysis - question features and non-answers 
xvi 
Table OA.1 – Question features and non-answers 
xvii 
Table OA.1 (cont.) – Question features and non-answers 
xviii 
Additional references 
xix 
 
 
 


## Page 2

Online Appendix 
ii 
 
OA 1 – GPT model overview and discussion 
There are many different GPT models available and new models come out frequently. This 
section covers model differences as well as an overview of popular models as of June 2024.   
OA 1.1 – Conceptual differences between GPT models 
From a practical point of view, there are five primary differences between GPT models:  
1. Size – The model size is commonly expressed by the number of parameters (i.e., the number 
of “coefficients” in the neural network) and roughly reflects the capabilities of a model. For 
example, a 70b (70 billion parameter) model is generally expected to perform better out-of-
the-box relative to a 7b (7 billion parameter) model. The downsides of model size are the 
memory requirements and compute time which scale with the number of parameters. Larger 
models require more expensive hardware and are generally slower.  
2. Tuning – The model tuning reflects how the model responds and the types of tasks it is good 
at. For example, the ChatGPT model is “instruct tuned” which means that it is designed to 
provide responses like a human assistant. Similarly, a model like CodeLlama-70b-Python is 
designed such that it is good at generating Python code. The tuning of a model is the product 
of two primary elements: the content of the original training data and the (optional) process 
with which it was fine-tuned at a second (or later) stage. For example, the Qwen1.5-7B-Chat 
model by Alibaba is bilingual in English and Mandarin due to the original training data and 
will respond conversationally like a human assistant (indicated by the ”chat” part of the 
name) due to a second stage fine-tune that steers the model to respond like an assistant.  
3. Context length – The context length of a model reflects the maximum number of tokens that 
the model can handle at a time. The context length is measured as the total number of tokens 
in the prompt as well as the completion and is sometimes indicated in the model name. For 
example, the “16k” part of “gpt-3.5-turbo-16k” by OpenAI indicates that the model has a 
context length of 16,385 tokens. Practically, the context length determines how long of a 
document the model can handle before it throws an error. Models with longer context lengths 
can handle longer input documents, although they might not pay equal attention to all parts of 
long documents. For example, Liu et al. (2023) find that GLLMs tend to focus primarily on 
information at the start and end of the context window.   
4. Availability – GPT models can be made available in multiple ways. The biggest distinction 
is whether the model weights are made available or whether the model is only available from 
behind a third-party API. For example, the model weights for popular models by OpenAI, 
Google, and Cohere (e.g., ChatGPT, GPT-4, Bard, Claude2) are not public, and you can only 
interact with each model through its respective API. Other models like Llama2 by Meta and 
Phi-2 by Microsoft have publicly available model weights, often through the Huggingface 
platform. These “open source” models can be downloaded and run locally or in the cloud 
with access to the appropriate hardware. Some open-source models do have restrictive 
licenses that, for example, prevent them from being used for commercial purposes. 


## Page 3

Online Appendix 
iii 
 
5. Quantization – Some models are quantized. Model quantization means that the model 
weights are rounded to have a lower precision (i.e., “have fewer decimal places”). Reducing 
the precision to half-precision (16-bit), 1/4th precision (8-bit), or 1/8th precision (4-bit) helps 
to reduce the memory requirements necessary to run the model. For example, an 8-bit 
quantized model requires only 1/4th the memory relative to its full precision equivalent.  
Quantized models are quicker and cheaper to use although the quantization process can result 
in a drop in performance relative to the full precision model.  
OA 1.2 – Which models to try first for your task? 
When considering GLLMs for a research project, I recommend starting with the flowchart in 
Figure 1 of the paper. If a third-party API is possible and feasible then I recommend starting with 
the OpenAI API and the ChatGPT or GPT-4 models. These models are competitively priced for 
the performance they deliver out of the box. The OpenAI models are also available through 
Microsoft Azure, which can make for easier reimbursement. The Google models are what I 
recommend trying second among the API providers, especially once the new Gemini 1.5 
becomes more generally available. When considering local models, I recommend starting with a 
Llama model or a derivative, such as Llama3 70b or Mistral 8x7b for zero-shot classifications. 
For local model fine-tuning, I recommend trying Phi-3 mini, Mistral 7b, or one of the Qwen 
models as they have good fine-tuning support (e.g., by Llama-Factory). The companion website 
provides code examples on how to get started with the OpenAI API as well as with local models.  
Finally, I also recommend doing some initial experiments with the Tiktoken package to get a 
sense of the context window requirements for your task. This will help navigate the table below. 
For example, a task with long inputs that are up to 20k tokens long will rule out many models.  
OA 1.3 – Overview of popular GPT models 
Provider 
Model 
Size / 
Cost 
Context 
length 
Availability 
Fine-
tunable? 
Tuning 
OpenAI 
gpt-3.5-turbo 
?? / $$1 
4k 
API2 
Yes 
Chat assistant 
OpenAI 
gpt-3.5-turbo-
16k 
?? / $$ 
16k 
API 
Yes 
Chat assistant 
OpenAI 
gpt-4-turbo  
& gpt-4o 
?? / $$$ 
128k 
API 
No 
Chat assistant 
OpenAI 
gpt-4 
?? / $$$$ 
8k 
API 
No 
Chat assistant 
OpenAI 
gpt-4-32k 
?? / $$$$ 
32k 
API 
No 
Chat assistant 
OpenAI 
GPT base 
babbage3 
?? / $$ 
16k 
API 
Yes 
Text completion 
 
1 The model size of proprietary models is often not public, but the API pricing provides a hint about the size of the 
model. I have added a dollar indicator to reflect that, where $ is a small model and $$$$ is a very large model.  
2 OpenAI models are available both through the OpenAI API as well as through Microsoft Azure.  
3 Note that these “GPT base” models replace the original GPT-3 models which are no longer available. 


## Page 4

Online Appendix 
iv 
 
OpenAI 
GPT base 
davinci 
?? / $$$ 
16k 
API 
Yes 
Text completion 
Anthropic 
Claude 2.0 
?? / $$$ 
100k 
API 
Yes 
Chat assistant 
Anthropic 
Claude 2.1 
?? / $$$ 
200k 
API 
Yes 
Chat assistant 
Anthropic 
Claude Instant 
?? / $$ 
100k 
API 
Yes 
Chat assistant 
Google 
Gemini 1.5 
?? / $$$$ 
128k/1m4 
API 
No 
Chat assistant 
Google 
Gemini 1.0 
Ultra 
?? / $$$ 
32k 
API 
No 
Chat assistant 
Google 
Gemini 1.0  
Pro 
?? / $$ 
32k 
API 
No 
Chat assistant 
Google 
Gemma 2b 
2b / $ 
8k 
Local 
Yes 
Text completion  
or chat assistant 
Google 
Gemma 7b 
7b / $ 
8k 
Local 
Yes 
Text completion  
or chat assistant 
Meta 
Llama2/3 7b 
7b / $ 
4k / 8k 
Local 
Yes 
Text completion  
or chat assistant 
Meta 
Llama2 13b 
13b / $$ 
4k 
Local 
Yes 
Text completion  
or chat assistant 
Meta 
Llama2/3 70b 
70b / $$$ 
4k / 8k 
Local 
Yes 
Text completion  
or chat assistant 
Microsoft 
Phi-2 
2.7b / $ 
2k 
Local 
Yes 
Chat assistant 
Microsoft 
Phi-3 mini 
3.8b / $ 
128k 
Local 
Yes 
Chat assistant 
Mistral 
Mistral 7b 
7b / $ 
8k 
Local 
Yes 
Text completion  
or chat assistant 
Mistral 
Mistral 8x7b5 
45B / $$$ 
32k 
Local 
Yes 
Text completion  
or chat assistant 
Alibaba 
Qwen 1.8b 
1.8b / $ 
32k 
Local 
Yes 
Chat assistant, 
English+Chinese  
Alibaba 
Qwen 7b 
7b / $ 
32k 
Local 
Yes 
Chat assistant, 
English+Chinese 
Alibaba 
Qwen 14b 
14b / $$ 
32k 
Local 
Yes 
Chat assistant, 
English+Chinese 
Alibaba 
Qwen 72b 
72b / $$$ 
32k 
Local 
Yes 
Chat assistant, 
English+Chinese 
The open-source (“local”) models in the table above also have hundreds of derivative variants 
that aim to provide improvements such as different tuning or larger context lengths. For a good 
overview, I recommend checking out the following website: https://huggingface.co/models.  
 
4 The model has a 128k context window by default with a 1 million context window only for selected users.  
5 A sixth difference is the architecture design of the model, which can vary slightly between models. For example, 
the Mixtral 8x7b model is special in that it uses the Mixture of Experts (MOE) approach, which works by combining 
eight 7b models together into a single model. For most applications the architecture details do not matter, but they 
are worth keeping in mind as it might help explain errors or other behaviors you run into when using the model.  


## Page 5

Online Appendix 
v 
 
OA 2 – Full prompts 
OA 2.1 - ChatGPT zero shot (Table 1 column 3): 
In the provided text, the analyst posed the question:   
> {question}  
 
The manager responded with:    
> {answer} 
 
**Task 1 - sentences:** 
 
From the manager's response, extract only those sentences where the 
manager explicitly indicates either: 
 
- They currently lack specific details or information to provide. 
- They are deliberately choosing not to share specific information at the 
moment. 
 
Please exclude sentences that discuss general uncertainties, company 
plans, or any actions the company might take in the future.  
Also exclude disclaimers that are immediately followed by an answer. 
 
Important: these sentences are rare, in 65% of the cases this will be an 
empty list. It is ok to return an empty list. 
 
Provide your response using the following JSON format: JSON = {{ 
    "sentences" : [] 
}} 
 
JSON = 
OA 2.1 - GPT-4 zero shot (Table 1 column 4): 
In the provided text, the analyst posed the question:   
> {question}  
 
The manager responded with:    
> {answer} 
 
**Task 1 - sentences:** 
 
From the manager's response, extract only those sentences where the 
manager unambiguously indicates: 
 
1. A clear absence of specific details or information that prevents them 
from providing an answer. 
2. A conscious decision not to divulge particular information at the 
moment. 
 
Important criteria for valid sentences: 
 
- Must state a lack of knowledge about a specific detail. 
- Must clearly indicate a choice not to share a particular detail. 
 
Exclude: 


## Page 6

Online Appendix 
vi 
 
 
- Statements expressing general uncertainty or vagueness. 
- Sentences about company plans or potential future actions. 
- Disclaimers, deferrals to others, or any remarks that still convey a 
type of answer or opinion. 
 
**In essence, the qualifying sentence should firmly suggest: "I don't have 
that specific detail" or "I choose not to share that specific detail 
now."** 
 
Provide your response using the following JSON format: JSON = {{ 
    "sentences" : [] 
}} 
 
Examples of sentences to exclude / ignore: 
- "I'll let another person handle the specifics." 
- "But time will tell." 
- "We will decide that in the future." 
- "The exact details are yet to be decided." 
 
Be extremely selective. Only include sentences you are 100% sure about. 
 
JSON = 
OA 2.3 - ChatGPT zero shot filter (Table 1 – column 5) 
Analyst question: 
{question} 
 
Manager response: 
{answer} 
 
A research assistant has marked the above response as including a 
statement that reflects unwillingness or inability to answer (part) of the 
analysts' question, because of the following comment(s): 
 
> {comments} 
 
Based on the question and full response above, provide a detailed 
assessment 
whether 
the 
manager's 
response 
includes 
a 
statement, 
explanation, or justification indicating an inability or unwillingness to 
answer the question. If you classify the response as reflecting inability 
or unwillingness to answer, justify your classification with specific 
phrases or sentences from the manager's response. If there's no such 
indication, explain why not. 
 
Your response should be in the following JSON format: 
JSON = {{ 
    "assessment" : a detailed assessment unique to this evaluation 
following the instructions above, 
    "your_classification" : 1 or 0 
}} 
JSON =  
OA 2.4 - GPT-4 zero shot filter with dimensions (Table 3, columns 1 to 3) 


## Page 7

Online Appendix 
vii 
 
In the provided text, the analyst posed the question:   
> {question}  
 
The manager responded with:    
> {answer} 
 
**Task 1 - sentences:** 
 
From the manager's response, extract only those sentences where the 
manager explicitly indicates either: 
 
- They currently lack specific details or information to provide. 
- They are deliberately choosing not to share specific information at the 
moment. 
 
Please exclude sentences that discuss general uncertainties, company 
plans, or any actions the company might take in the future. 
 
**Task 2 - reasons:** 
 
Why is the manager reluctant, unable, or unwilling to share the specific 
information?  
Your choices:  
- dont_know -> the manager does not know or have the numbers and/or it is 
too early to tell 
- cant_give -> the manager states that they don't provide that information 
as per policy, norm, or principle 
- proprietary -> the manager states that the information is proprietary or 
sensitive for competitors  
- invalid_match -> the answer is incorrectly flagged as containing a 
refusal or hesitation to answer, select if no sentences. 
 
**Task 3 - alternative:** 
 
Does the manager still provide an alternative answer, if so what do they 
provide? Your choices:  
 
- range_or_perc -> the manager instead provides a range estimate or 
another type of quantitative estimate such as a percentage 
- comparison -> the manager instead makes a quantitative or qualitative 
comparison to some other point of reference 
- qualitative -> the manager instead provides a qualitative statement 
without sharing exact details or numbers 
- no_alternative -> the manager provides no further information or only 
explains their reasoning for not answering 
- invalid_match -> the answer is incorrectly flagged as containing a 
refusal or hesitation to answer, select if no sentences. 
 
**Task 4 - topic:** 
 
What is the part of the question about that the manager is not, or only 
partially, answering? Your choices: 
 


## Page 8

Online Appendix 
viii 
 
- forward_looking - the manager refuses to provide guidance or other 
forward-looking information 
- breakdown - the manager refuses to provide a breakdown or more 
granularity or detail on something 
- rd_clinical_trails - the manager refuses to provide more information on 
a research & development projects or clinical trails  
- regulation_certification - the manager refuses to speak about a 
regulatory matter, a compliance issue, or an ongoing certification by a 
third-party 
- customer_or_supplier - the manager refuses to provide details on major 
customers or suppliers 
- major_investment - the manager refuses to speak about a merger and/or 
acquisition or an other type of major investment, collaboration, or 
partnership 
- other  
 
**Task 5 - sentiment:** 
 
In the manager response, are they trying to paint a positive, negative, or 
neutral picture?  
 
- optimistic -> the manager is trying to paint a (somewhat) positive 
picture with their alternate answer 
- pessimistic -> the manager is trying to paint a (somewhat) cautionary or 
pessimistic picture with their alternative answer 
- neutral -> the manager is painting is trying to paint a neutral picture 
 
Provide your response using the following JSON format: JSON = {{ 
    "sentences" : [], 
    "reasons-discussion" : a brief discussion of your choice for 
"reasons", 
    "reasons" : dont_know / cant_give / properietary / invalid_match, 
    "alternative-discussion" : a brief discussion of your choice for 
"alternative", 
    "alternative" 
: 
range_or_perc 
/ 
comparison 
/ 
qualitative 
/ 
no_alternative / invalid_match, 
    "topic-discussion" : a brief discussion of your choice for "topic", 
    "topic" 
: 
forward_looking 
/ 
breakdown 
/ 
rd_clinical_trails 
/ 
regulation_certification / customer_or_supplier / major_investment, 
    "sentiment-discussion" : a brief discussion of your choice for 
"sentiment" 
    "sentiment" : optimistic / pessimistic / neutral 
}} 
 
JSON = 
OA 2.5 - ChatGPT fine-tuned filter with dimensions (Table 3, columns 4 to 6)  
Analyst question: 
> {question}  
 
Manager response: 
> {answer} 
#### 
{json} <|end|> 
 


## Page 9

Online Appendix 
ix 
 
OA 3 – Full description of the GPT method  
This section provides more detail on the four steps of the GPT method. This approach 
demonstrates the various ways to use GLLMs, which makes it more extensive than necessary. A 
two-step approach (e.g., keyword filter + fine-tuned GLLM) is sufficient for most use cases.  
Step 1: Keyword filter 
I first narrow down the Q&A pairs to those that possibly include a non-answer. Identifying 
possible non-answers is an easier task than identifying exact non-answers. Specifically, any false 
positives (i.e., sentences matched as a possible non-answer, but are not) will be taken care of in 
the later stages. So, the objective is to reduce the sample of possible non-answers while 
minimizing false negatives. Minimizing false negatives is the primary concern as any Q&A pairs 
without a keyword match will be marked as an answer and not be analyzed by the more powerful 
methods later.  
To develop the keyword list, I use the words in the regular expressions by Gow et al. (2021) as a 
start. I then manually extend this list with unigrams, bigrams, or trigrams that help reduce false 
negatives. The full keyword list is shown below. I apply the keyword search at the sentence level 
and mark the Q&A pairs with one or more keyword matches for further processing.  
Trigrams: 
call_it_out, at_this_time, at_this_point, at_this_moment, break_it_out, don_t_have, don_t_know 
Bigrams: 
not_going, will_not, won_t, by_region, get_into, that_level, are_not, don_t, do_not, give_you, 
break_out, splice_out, tell_you, too_early, can_t, can_not, not_ready, right_now, no_idea, 
not_give, not_sure, wouldn_t, haven_t 
Unigrams: 
cannot, comment, commenting, comments, unable, guidance, guide, guiding, forward, hard, talk, 
range, disclose, report, privately, forecast, forecasts, forecasting, specific, specifics, detail, 
details, public, publicly, provide, breakout, statement, statements, update, announcement, 
announcements, answer, answers, quantify, share, sharing, information, discuss, mention, sorry, 
apologies, apologize, recall, remember, without, specifically, difficult, officially 
Step 2: Basic machine learning filter 
The keyword approach is simple and fast, but it is naïve in its implementation as it does not take 
any context into account. An alternative filtering approach is to use a basic supervised machine 
learning algorithm. To demonstrate that, I next train a classifier using the OpenAI API to classify 
sentences that possibly contain a non-answer. Like in step 1, the objective is to reduce the sample 
while minimizing false negatives. To train the classifier, I randomly draw a sample of 1,500 
Q&A pairs and classify the resulting 2,100 response sentences that contain a keyword match. 
The performance statistics show a non-answer recall score of 97%, which indicates a minimal 
number of false negatives. I apply this classifier to the keyword-matched sentences and retain 
Q&A pairs with one or more sentence matches for further processing. 


## Page 10

Online Appendix 
x 
 
The prompt and completion templates for the GPT-3 Babbage model are simple and shown 
below: 
prompt_template = """ <|input|> {sentence} -> """ 
completion_template = """ {answer - 0/1} <|end|> """ 
Step 3: Zero-shot ChatGPT 
The two filtering steps reduce the sample of Q&A pairs with possible non-answers to about 1/3rd 
of the full sample. These remaining pairs contain the answers that are hardest to classify as they 
all have some element of a non-answer to them. I demonstrate two ways, zero-shot and fine-
tuning, of using a ChatGPT model to solve the remaining classifications. These final 
classifications are done at the Q&A pair level, not the sentence level, to provide the model with 
the full question-and-answer context when making the non-answer classifications.  
For the zero-shot ChatGPT approach (i.e., step 3) I design the prompt shown in OA 2.3. The 
zero-shot prompt has a few notable design considerations. First, I am only applying this prompt 
on the sample of Q&A pairs that remain after the filtering step, which makes it a distinct task 
from the full sample zero-shot prompt in OA 2.1. Second, I help ChatGPT by providing it with 
the matched sentences from steps 1 and 2 (i.e., the “comments” portion in the prompt). Third, I 
frame the task in terms of “evaluate the work of a research assistant”, to help nudge ChatGPT to 
make more detailed and nuanced judgments for trickier cases where the classification is not 
immediately obvious. Finally, I apply the chain-of-thought technique to improve classifications 
by asking ChatGPT to first make a detailed assessment before its final classification. I feed all 
the Q&A pairs that remain after Steps 1 and 2 through this zero-shot prompt and extract the non-
answer classifications from the completions.  
Step 4: Fine-tuning ChatGPT 
The zero-shot ChatGPT approach works well, although it sometimes overestimates the 
probability of a non-answer. This is a challenge with zero-shot approaches as it can be hard to 
convey complex concept definitions and their distributional properties using only a short prompt. 
Fine-tuning methods can help with that, as they convey the instructions through a sample of 
examples rather than prompt instructions. So, to further improve the classifications coming out 
of step 3, I feed all the classifications through a ChatGPT (GPT 3.5) model that is fine-tuned on a 
training sample that I construct using GPT-4 and manual classification. As part of this fine-
tuning step, I also classify four additional dimensions of non-answers with little additional costs. 
Classifying the additional dimensions helps to demonstrate the flexibility of GLLMs and 
provides additional data to improve our understanding of earnings call non-answers. An 
overview of the different non-answers dimensions is shown in Appendix A of the main paper.  
I start by developing a zero-shot GPT-4 prompt to create a start for the training sample. The 
resulting prompt is shown in OA 2.4. The zero-shot prompt has a few notable design 
considerations. First, I manually identify common types of false positives by step 3 (such as 
statements about not pursuing company actions being flagged as non-answers) and tailor the 
GPT-4 prompt to specifically filter these out. Bringing in plenty of task-specific context 
generally helps improve performance with the biggest models like GPT-4, which can handle the 


## Page 11

Online Appendix 
xi 
 
additional detail. Second, instead of a 0 / 1 classification, I require the model to identify the exact 
non-answer phrases. This is more challenging but, in my testing, increasing the complexity of the 
task can sometimes reduce outcome variability and improve performance. Afterward, I transform 
the matched sentences into a 0 / 1 variable, where no sentences receive a 0. Third, the definitions 
for each of the dimensions have a degree of ambiguity to them. To help communicate to the 
model what I am after I spell out not only the possible options but also additional context for 
each of the options. Finally, I ask the model to make five classifications through a single prompt. 
Doing so makes performance slightly harder to evaluate, however, in my testing it improves 
overall performance relative to creating a separate prompt for each classification. Including all 
the questions at once seems to help GPT-4 understand what a non-answer is and what is not.  
After developing the prompt, I draw a random sample of 400 Q&A pairs that likely contain a 
non-answer (i.e., those remaining after the Column 5 approach in Table 1). The performance 
results in Table 3 of the main paper show that the GPT-4 zero-shot approach works well but has 
two downsides: it is not cost-efficient for large samples and it remains hard to communicate what 
exactly constitutes a non-answer. To help with that, I demonstrate a hybrid approach to fine-
tuning a ChatGPT model on a training dataset created using zero-shot GPT-4. Specifically, I start 
by drawing another random sample of ~2,100 Q&A pairs and feed them through the zero-shot 
GPT-4 approach. The performance statistics in Table 3 show that GPT-4 achieves near-human 
performance for the four sub-dimensions, so I directly use those for the training data. The non-
answer classification by GPT-4 can use improvement, so I manually go through all the training 
observations and improve the classification of non-answers with my own. I then use the 
improved training dataset and the OpenAI API to fine-tune a ChatGPT (GPT 3.5) model using a 
reduced form of the GPT-4 prompt, which is shown in OA 2.5. It is worth pointing out that the 
fine-tuning prompt only includes the question, the answer, and the JSON completion. We do not 
have to include the instructions as we communicate to the model what we want through our 
training data examples and not through prompt instructions. As shown in Table 3, the mean 
tokens per Q&A pair drop from 1,267 tokens for the zero-shot GPT-4 prompt to 324 tokens for 
the ChatGPT fine-tuned prompt. The lower token counts result in a substantial cost saving in 
addition to the already lower price of using a fine-tuned ChatGPT model versus a GPT-4 model.  
 
 


## Page 12

Online Appendix 
xii 
 
OA 4 – Examples of classification differences between the GPT and Gow et al. methods 
Below are fifteen random examples for each area of the Venn diagram in Figure 2. The Gow et 
al. (2021) regular expression matches are shown in bold. Comparing the error rates in the left 
versus right slice highlights the considerably lower non-answer error rate by the GPT method.  
Left slice – The Gow et al. method marks it as a non-answer but the GPT method does not: 
• 
So, not a lot of price increases relative to just supply and demand right now, really more about 
driving technology innovation to keep our price points higher, and reflect the value of going into 
those customers. 
• 
And I think the idea is to try to not give anyone a heart attack in terms of a big change in our 
structure. 
• 
It’s hard when we talk about the back half of the year. 
• 
We need to execute this, it’s not, for us to get material revenue growth, I don’t have to deal with 
Metronic I just have to execute on the relationships that the company has already put in place. 
• 
Looking at a conversion lift in the Non-Hotel category from the additional brand awareness, next to 
impossible for us to tell, in part because those components on TripAdvisor are growing so strong 
all by themselves. 
• 
We are – when we do our monthly financial performance reviews, we are seeing that we’re coming in 
under cost for the quarter and that’s not too surprising given the activity and we have a lot of 
remote work. 
• 
That -- I don't know that it was those orders specifically. 
• 
I don't have that number off the top of my head. 
• 
We just don't know we're looking at everything look, we've looking at our capital allocation, we're 
applying it in different things, we've got some club units that we think we can you know depending 
on if we can find a buyer for them, we may sell a couple of club units. 
• 
There is no question of profits shattered out there and I’m not going to comment on the shatter but 
what I will tell you is we have created a very unique business with tremendous opportunities for 
growth as we watch consumer demand for snacks growing at a very healthy rate around the world, 
and we capitalize on our strong positions, especially in these emerging markets. 
• 
We have new products or customers have changed circumstances such as consolidations, so we're 
managing a dynamic set of relationships, but fundamentally our duration, I don't know, the average 
number is maybe about 2.7 years or 2.8 years or so, and so that has been remarkably confident. 
• 
So I remember, and I'm not going to name -- but I remember talking so many CEOs and their 
number one objective was, oh, I have to start being able to people text their orders into the store. 
• 
I don't think we have that. 
• 
So we're more in the mode of, as we're trying to play through, as I've referred to with the whack-a-
mole analogy is, you get some buildup, because you might have built up your components to build an 
assembly or package of sorts and then you don't get a component. 
• 
So we're not -- have any supply issues to doing that or supply chain issues. 
Overlap area – Both methods mark the response as a non-answer:  
• 
Now, you should not expect us to be given a volume. 
• 
Yes. We haven't provided that, Tom. Perhaps we'll provide it in the next quarter or later on. But right 
now, we have not provided the range. 
• 
We don't have specific granular views nor would we give those as part of the guidance. 


## Page 13

Online Appendix 
xiii 
 
• 
We're not disclosing at this time what indications we are going to pursue or exclude. 
• 
Yeah, I don't know the very exact numbers, but a few quarters ago in our slide that we gave you a 
price chart which shows you the break down market intelligence cost by - customers and was roughly 
pretty evenly split between four categories. 
• 
And so, we don't -- we're not going to give you guidance as to how we see it into the entire year, 
• 
With regards to what the weather impact is going to be on the season, I will tell you about that on the 
November call, because I've never been able to figure it out how to predict it at this point. 
• 
We haven't provided any guidance with respect to timeframe that profitability. And we are 
going to do that at this time, we may do that in the future but we are not providing guidance on that at 
this time. 
• 
Well, there is never an easy answer to this, but I'll let Jeremy give it his best shot here. 
• 
And as well as we hope that the economic stagnation that we see ourselves in, probably more non-
U.S. than U.S., comes back to life, but it's hard to predict, we'll see. 
• 
Well, I'm not going to speculate on 2015 at this point, 
• 
To what extent, we don't forecast. 
• 
Yes, on the last part of that, I don't know that we've 100% scheduled every maintenance event for 
next year and want to be nimble and on our toes, don't anticipate any changes. 
• 
I honestly don't know. 
• 
We can't tell by transaction data if we're seeing any, but I think the closest we could come is we've 
run a regression around the correlation between off-premise comp sales behavior and on-premise 
comp sales behavior. 
Right slice – The GPT method marks it as a non-answer but the Gow et al. method does not: 
• 
I would hesitate to get into forecasting specific items across the company, but you see the impact in 
the first quarter from volume and the extra costs related to unabsorbed fixed. 
• 
They don't share individual guests contact information. 
• 
I don't think we have enough there to know, Bill. 
• 
And we've got some testing to do and I really need you to let us get that testing done and come back 
and report on what we know and what we plan to do about it. 
• 
But then the other piece is we just – we are at only eight weeks into the air. So, that will need to make 
a change in terms of overall guidance. We'll continue to evaluate guidance as we go into 2019. 
• 
It is really early stages here on Vega. 
• 
So I don't have -- it's a long story, I don't have a number for you. 
• 
But on the door activation switch, we can't speak for the customer, but we hope they carry it over 
other platforms of course, but it is one award and we'll have to see what will happen. 
• 
Obviously, we have to finalize this JV agreement. So -- because that's not final, we have not included 
anything in the guidance. 
• 
Yes. I wish I had the crystal ball on that. 
• 
Yeah. I don't think we're sharing detailed stats on engagement by messaging platform. 
• 
It's very difficult at this point with the cycle and the level of volume that we're seeing to kind of 
outlook exactly when and where we'd feel that. 
• 
And as a result, we're very hesitant to call when the visibility will improve there. 
• 
And my comment to add to that is we don't have a specific number for you in terms of how that 
business across search, culture shaping and leadership consultants are going to breakdown. 
• 
It's hard to quantify what, why and where. 


## Page 14

Online Appendix 
xiv 
 
OA 5 - Using and fine-tuning a local GLLM  
Local GLLMs can provide a more private, replicable, and cost-effective alternative to third-party 
APIs. In this section, I provide an overview of local fine-tuning methods as well as anecdotal 
observations of replicating the non-answer GPT method using local GLLMs.  
OA 5.1 - Local GLLM fine-tuning 
Fine-tuning is an umbrella term to describe a set of methods to update the model weights of an 
LLM by means of a training sample. The fine-tuning methods differ on two main dimensions: 
(1) The parts of the model weights that are subject to updating.  
a. Full tuning is the original form of fine-tuning and allows all weights to be updated. 
This effectively continues the original training with a more specific training sample 
for your task. This is the most powerful but also the most computationally intensive.  
b. Parameter Efficient Fine Tuning (PEFT) are methods that allow fine-tuning while 
keeping all or most of the original model weights unchanged (i.e., frozen). This 
makes fine-tuning significantly easier but can come at the cost of performance.  
i. Partial or Freeze tuning are PEFT methods where most of the model weights are 
kept frozen and only the final model layers are allowed to update.  
ii. Low-Rank Adaptation (LoRA) tuning keeps the full model weights frozen and 
instead trains an adapter to change the model behavior (Hu et al. 2021). Combining 
the original model with your trained adapter then creates your fine-tuned model. 
Adapters are smaller in parameter size relative to the original model and thus 
easier to train. QLoRA approaches combine LoRA with quantization techniques.  
(2) The evaluation function used during fine-tuning to distinguish good from bad responses.  
a. Standard fine-tuning procedures have the same evaluation function as the original 
training procedure, which is to try to predict the same token as the original text. 
b. Reinforcement Learning from Human Feedback (RLHF) approaches use a different 
evaluation approach than the original training procedure. RLHF approaches evaluate 
the quality of a generation using a more sophisticated feedback mechanism. For 
example, ChatGPT is RLHF tuned by having human evaluators evaluate generations 
during various stages of the fine-tuning process, steering the model towards 
generations that are evaluated as higher quality by humans. Many RLHF approaches 
exist, notable mentions include Reward Modeling, Proximal Policy Optimization 
(PPO), and Direct Preference Optimization (DPO). RLHF tuning can yield better 
performance but is also much harder and more costly to implement.  
For textual analysis tasks, I recommend first trying a standard full fine-tune of a smaller model, 
such as Phi-2 2.7b or Llama2 7b. For most tasks, this will be the easiest way to get good 
performance. If that is not feasible or does not work, I next recommend trying to fine-tune a 
larger model using a LoRA adapter. These adapters make it possible to fine-tune larger models 
although it can be harder to get good results out of them relative to full fine-tuning. Finally, 
RLHF techniques are generally unnecessary when fine-tuning a model for textual analysis tasks.  


## Page 15

Online Appendix 
xv 
 
OA 5.2 – Non-answer replication using local GLLMs  
As an experiment, I replicate parts of the non-answer task using local GLLMs. I start by 
replicating the zero-shot results in Table 1 using an A6000 GPU, the Llama2 7b and 13b chat 
models, and the ChatGPT prompt shown in Online Appendix 2.3. Manual inspection of the result 
shows that the Llama2 models struggle to follow the instructions out of the box. The models 
frequently return text that is not JSON or JSON that is unparseable. Around 16% (8%) of the 500 
evaluations are unparseable with the 7b (13b) model. Furthermore, the remaining classifications 
are not good. For example, the 13b model marks 86% of the answers as containing a non-answer. 
However, there are a few important caveats to these results: detecting non-answers is an 
unusually hard task and I did not adapt the prompts to the Llama 2 models. Further prompt 
engineering would likely improve the results. The primary insight here is that zero-shot 
approaches using local GLLMs are generally more challenging to get working relative to zero-
shot approaches with third-party models such as ChatGPT and GPT-4.  
Fine-tuning is where local GLLMs tend to have the most potential. To demonstrate that, I next 
replicate the ChatGPT fine-tuning step in Table 3 by training a LoRA adapter for the Llama2 7b 
chat model using an RTX 4090. Fine-tuning the adapter took about an hour on the ~2,100 
training observations. The resulting fine-tuned model yields results that are significantly better 
than the zero-shot experiments. The model strictly follows the instructions and produces valid 
JSON, which was not possible in the zero-shot tests. Untabulated results show that the fine-tuned 
Llama2 7b model yields an F1 score of 0.77 for the non-answers, 0.90 for the non-answer 
justification, 0.77 for the non-answer type, 0.69 for the question type, and 0.83 for the non-
answer sentiment. Fine-tuning the Llama2 7b model with full fine-tuning on more powerful 
hardware would likely improve these results even further. Importantly, these results are getting 
close to those of the fine-tuned ChatGPT model, at 1/100th the cost. It only costs around $0.04 to 
generate 1,000 predictions using the fine-tuned Llama2 7b model, versus $4.18 for the same 
1,000 predictions using the fine-tuned ChatGPT model with the 2023 OpenAI API prices.  
 
 


## Page 16

Online Appendix 
xvi 
 
OA 6 – Additional analysis - question features and non-answers  
As an additional analysis, I also replicate Table 2 of Gow et al. (2021) to study how question 
characteristics relate to the propensity of a non-answer. Following Gow et al. (2021), I regress 
each question characteristic separately on the non-answer outcome with earnings call fixed 
effects. To save space, the results of the independent regressions are shown below each other in 
Table OA.1 Panel B, columns 1 and 2. There is one notable difference between my measure and 
the Gow et al. (2021) measure. Specifically, Gow et al. (2021) find a positive correlation 
between negative question sentiment and the likelihood of a non-answer. In contrast, the more 
accurate GPT measure finds a negative correlation with both positive as well as negative 
question sentiment. In both cases the question sentiment is calculated using the Loughran & 
McDonald (2011) wordlists, the only difference is the non-answer classification. These new 
results show that managers are also less likely to provide a non-answer when an analyst asks a 
question with a strong negative sentiment, although to a lesser degree than a question with a 
strong positive sentiment.  
Next, I also expand the analysis by correlating the same question features with the different types 
of non-answers. The results highlight that the question feature correlations are conditional on the 
type of non-answer. A few notable results. First, having a question with a strong positive 
sentiment does not impact the likelihood of managers giving a positive spin to their non-answer 
(-1.666, Column 4). A question with a strong negative sentiment, on the other hand, does 
significantly decrease the likelihood of the manager giving a positive spin (-24.240***, Column 
4). Second, question complexity (i.e., Q - Fog index) generally predicts a higher likelihood of a 
non-answer. However, this is driven by non-answers with a qualitative alternative (0.196***, 
Column 6). Complete refusal and range or percentage non-answers instead are less likely for 
complex questions (-0.050***, Column 5 and -0.030**, Column 7). Finally, questions that ask 
about specific amounts, locations, or dates are generally more likely to receive a non-answer. 
However, these categories are significantly less likely to receive a complete refusal and more 
likely to receive a qualitative non-answer (-0.191***, -0.291***, -0.146*** in Column 5).  


## Page 17

Online Appendix 
xvii 
 
Table OA.1 – Question features and non-answers 
Panel A: Descriptive statistics 
 
(1) 
(2) 
(3) 
(4) 
(5) 
(6) 
(7) 
(8) 
 
Obs. 
Mean 
Std. 
Dev 
P5 
P25 
P50 
P75 
P95 
 
 
 
 
 
  
  
 
 
Question features 
 
 
 
 
 
 
 
 
Positive tone 
1,152,473 
0.011 
0.02 
0.00 
0.00 
0.00 
0.02 
0.04 
Negative tone 
1,152,473 
0.011 
0.02 
0.00 
0.00 
0.00 
0.02 
0.04 
Uncertainty 
1,152,473 
0.014 
0.02 
0.00 
0.00 
0.01 
0.02 
0.05 
Fog index 
1,152,473 
10.23 
4.55 
4.33 
7.50 
9.61 
12.14 
18.07 
Contains numbers 
1,152,473 
0.394 
0.49 
0.00 
0.00 
0.00 
1.00 
1.00 
Contains money or percent 
1,152,473 
0.160 
0.37 
0.00 
0.00 
0.00 
0.00 
1.00 
Contains location 
1,152,473 
0.136 
0.34 
0.00 
0.00 
0.00 
0.00 
1.00 
Contains date or time 
1,152,473 
0.544 
0.50 
0.00 
0.00 
1.00 
1.00 
1.00 
 
 
 
 


## Page 18

Table OA.1 (cont.) – Question features and non-answers 
Panel B: Determinant results 
  
(1) 
(2) 
(3) 
(4) 
(5) 
(6) 
(7) 
  
Non-answer 
GPT approach 
Non-answer 
Gow et al.  
Non-answer  
Can't give 
Non-answer  
Positive 
Non-answer  
Complete 
refusal 
Non-answer  
Qualitative 
Non-answer  
Range or Perc. 
 
 
 
 
 
  
  
  
Q - Positive sentiment 
-48.598*** 
-34.180*** 
-14.813*** 
-1.666 
-14.421*** 
-22.937*** 
-11.240*** 
(3.019) 
(4.062) 
(0.641) 
(3.764) 
(0.844) 
(3.700) 
(0.432) 
Q - Negative sentiment 
-14.763*** 
6.892** 
-3.505* 
-24.240*** 
-1.209 
-13.947*** 
0.394 
(3.020) 
(2.788) 
(1.761) 
(1.455) 
(1.147) 
(1.680) 
(0.628) 
Q - Uncertainty 
70.868*** 
50.331*** 
27.300*** 
42.506*** 
10.081*** 
50.329*** 
10.458*** 
(3.754) 
(3.376) 
(0.911) 
(3.075) 
(1.299) 
(3.391) 
(0.870) 
Q - Fog index 
0.116*** 
0.226*** 
0.010* 
0.170*** 
-0.050*** 
0.196*** 
-0.030*** 
(0.013) 
(0.020) 
(0.005) 
(0.007) 
(0.006) 
(0.010) 
(0.002) 
Q - Contains numbers 
3.262*** 
2.302*** 
0.770*** 
2.149*** 
0.008 
2.302*** 
0.952*** 
(0.269) 
(0.191) 
(0.098) 
(0.203) 
(0.036) 
(0.212) 
(0.045) 
Q - Contains money 
1.550*** 
1.347*** 
0.487*** 
0.981*** 
-0.191*** 
0.724** 
1.017*** 
(0.290) 
(0.159) 
(0.102) 
(0.230) 
(0.056) 
(0.235) 
(0.052) 
Q - Location 
0.745*** 
1.074*** 
-0.034 
0.867*** 
-0.291*** 
1.127*** 
-0.091** 
(0.159) 
(0.096) 
(0.046) 
(0.091) 
(0.060) 
(0.111) 
(0.038) 
Q - Date or time 
3.299*** 
2.725*** 
0.999*** 
2.455*** 
-0.146*** 
2.703*** 
0.742*** 
(0.163) 
(0.174) 
(0.063) 
(0.148) 
(0.025) 
(0.130) 
(0.038) 
Call FE 
Yes 
Yes 
Yes 
Yes 
Yes 
Yes 
Yes 
Multivariate 
No 
No 
No 
No 
No 
No 
No 
N 
1,152,473 
1,152,473 
1,152,473 
1,152,473 
1,152,473 
1,152,473 
1,152,473 
  
  
  
  
  
  
  
  
 
 
 


## Page 19

Additional references 
Hu, Edward J., Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu 
Wang, and Weizhu Chen. 2021. “LoRA: Low-Rank Adaptation of Large Language 
Models.” arXiv. 
Liu, Nelson F., Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, 
and Percy Liang. 2023. “Lost in the Middle: How Language Models Use Long Contexts.” 
arXiv. 
Loughran, Tim, and Bill Mcdonald. 2011. “When Is a Liability Not a Liability? Textual 
Analysis, Dictionaries, and 10-Ks.” The Journal of Finance 66 (1): 35–65. 
 


