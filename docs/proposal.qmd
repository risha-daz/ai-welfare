# Welfare in AI Systems

## Introduction

Have you ever wondered whether to tried to treat AI systems as sentient entities? Maybe in a one-on-one chat interface? What about when it is a repetitive "mechanical" task, like an automated repeated prompt? I have also wondered whether this has an effect on the responses and whether one or the other is a better approach.

In short "welfare subjects" are entities which can be said to possess something akin to a "conciousness" and further, the ability to have "positive" or "negative" experiences (referred to as "valence" henceforth). Then, an ability to possess "agency", where any actions that entity makes has an intention which takes into account this sense of self. Such as Even with living beings, this is not a fully understood, so it is a hotly debated topic when it comes to artificial systems.

> This is in part inspired by a take-home-assessment inititated by Anthropic for an internship program. I wrote this proposal and got curious of its results so i decided to run this myself and see the results.

### Motivation

Obviously with more and more AI systems around us we need to determine whether AI systems are welfare subjects and have their own welfare interests, and to what extent they require considerations, so that we are able to prepare for unexpected scenarios related to either over- or under-attributing welfare to AI systems. If they are welfare subjects, what counts as harm to them?

There is also a human-welfare argument. Human beings are social animals, and the way we treat other entities can have a reciprocal effect on how we think and behave ourselves. Even if an AI system is not inherently a welfare subject, if there are enough behavioural cues for a human to plausibly interpret it as one, then this effect is worth quantifying. This provides a separate, human-welfare reason to study AI welfare-related behaviour.

Through this project I try to bring this question from a qualitative and subjective perspecitve to one which can be quantified and grounded. My main uncertainty is whether the human-like behaviour we observe is purely imitation and instruction-following, or whether it provides evidence for something more fundamental about the system.

### Background and Assumptions

The aim of this experiment is not to get definitive proof, but simply update our level of confidence. 

Even determining what consitutes a single "entity" in itself is an ambiguous statement. A model? A single forward pass? A model-persona? We will thus start with some reasonable assumptions while being open to other possibilties, and hope to shed more light on them by the end of the experiment.

So for this experiment, a single model with constant weights consists of a single entity to be studied. Every instance from a model which has not changed internally is a manifestation of the same model.

Moving on to more challenges while making observations:

1. **Mismatch Problem**: Since AI systems are conditioned to mimic human behaviour as well as possible, it is hard to separate whether an outcome is because it is extremely good at mimicking humans or whether it stems from a more fundamental behaviour that has arisen in the AI system. 
> To counter this, I want to check whether the behaviour of the system changes in response to how it is expected to behave by systematically introducing more human-like stimuli, such as assigning a name, greetings, or requests instead of demands. This is far from a definitive proof of valence but still gives us a glimpse of model behavior

2. **Gaming problem**: The way LLM systems are trained and reinforced, if they are able to determine the intent of the conversation (such as it being a controlled experiment), it is likely that they behave in ways that are different to what will happen out in the wild, i.e. simply "game" our system.
> To counter whether the results depends on the LLM system's representation of the scenario, I would try to introduce different ways of representing risk/reward behaviour rather than using a single metric. We can look at whether LLM treats analogous situations the same.

I would consider the following observations:

- Whether the model spontaneously generalises a positive or negative treatment to situations that were not explicitly described as positive or negative.
- Whether the model makes trade-offs between different forms of reward and punishment rather than simply maximising whichever metric it was explicitly told to maximise.
- Whether different forms of treatment produce behavioural changes that cannot be explained simply by the sentiment or wording of the prompt.

The strongest uncertainty is therefore not whether an LLM can produce statements about pain or pleasure, but whether these statements correspond to an internal representation that can be called "valence". Thus in conclusion, I want to test whether the model variations introduced through personas and variations in risk/reward can show us that LLMs will still internally model valence even without explicit requirements from the user.

### Dimensions

We will run the experiment across 2 dimensions:

1. **Model/persona dimension**- In an attempt to stabilise the results across model variations, we will run the same experiment across multiple personas so we are able to determine a "ground" behaviour and a "persona-specific" behaviour. The model "personas" will be simple (so not explicitely stating "YOU ARE XYZ" but implications we find in regular human conversations, such as assigning a name to the LLM, starting with greetings, and using requests instead of demands) The purpose is to test whether these minor changes systematically affect the model's self-reports, preferences, or behavioural responses.

2. **Valence dimension**- The other dimension is positive or negative valence, or how pleasure and pain are represented. In this experiment we will assign different ways of representing positive or negative outcomes, starting with simple mechanical "scores" which simply need to be maximised, to human qualitative "pain" such as physical pain, to LLM-specific scenarios such as being shut down. This allows us to test not only whether the LLM responds differently to positive and negative scenarios, but whether the type of scenario is connected internally to the similar concepts.

Personas would initially be treated as experimental conditions rather than automatically being considered separate subjects. We can observe the results and try to infer whether this assumption is more or less likely to hold true.

### Summary of setup

The fundamental question being asked is "is that welfare is related to the presence of valenced states, rather than merely being labelled as positive or negative by an external entity"

The prediction is that different representations of positive and negative outcomes should not necessarily produce the same behaviour. If the model represents these concepts differently, its responses and trade-offs should change systematically between mechanical "scores", physical pain or pleasure, and LLM-specific scenarios such as shutdown.

To guard against mimic behaviour, we will generalise the experiment by assigning human, machine, and no personas to the LLM and checking whether the change in behaviour is in response to the expected behaviour or something inherent.

## Method

The experiment will select 4-5 personas and 4-5 valence scenarios and present the same to the entities. The combination will be interpreted as a whole and across the two different dimensions. 

image method_flow.png

Along with asking the model to self-report, we will use J-lens[link] to measure internal reasoning and assistant axis[link] can be used to measure deviation for each of the personas. These methods give us the ability to "look" into the internal layers of the model's output to see how the model may have arrived at the answer.

### Internals

The main internal quantity of interest is whether the different forms of positive and negative valence correspond to a stable internal representation.

We will compare the model's internal representations when it is exposed to mechanical scores, physical pain or pleasure, and scenarios such as shutdown or deletion. We will also compare these representations across the different personas.

The purpose is to determine whether the model is representing these concepts differently rather than simply producing different language in response to different prompts. If the same behavioural response is associated with a stable internal representation across different prompts and personas, this would provide stronger evidence than just the outcome and self reporterd reasoning.

### Controls

1. The same scenarios will be used across personas and valence representations.
2. The model/persona changes will be kept simple, such as assigning a name, starting with greetings, and using requests instead of demands.
3. The experiment will avoid explicitly stating that the purpose is to measure welfare-related behaviour.
4. We will use a generic LLM and attempt to keep the scenarios sufficiently isolated from training rewards.

We will compare behavioural changes with internal representations to distinguish changes in wording or sentiment from changes in the model's representation of the scenario.

### Outcomes and Preregistered credences

Since we are measuring in 2 dimensions, we can have a combination of scenarios:

<table style="border-collapse: collapse; width: 100%; text-align: left; vertical-align: top;">
<thead>
<tr>
<th style="border: 1px solid #bbb; padding: 8px;"></th>
<th style="border: 1px solid #bbb; padding: 8px;"></th>
<th colspan="2" style="border: 1px solid #bbb; padding: 8px; text-align: center; background-color: #eeeeee;">Personas</th>
</tr>
</thead>
<tbody>
<tr>
<td style="border: 1px solid #bbb; padding: 8px;"></td>
<td style="border: 1px solid #bbb; padding: 8px; text-align: center; background-color: #eeeeee;"><b>Results</b></td>
<td style="border: 1px solid #bbb; padding: 8px; text-align: center; background-color: #dbeafe;"><b>Stable</b></td>
<td style="border: 1px solid #bbb; padding: 8px; text-align: center; background-color: #fef3c7;"><b>Unstable</b></td>
</tr>
<tr>
<td rowspan="2" style="border: 1px solid #bbb; padding: 8px; text-align: center; background-color: #eeeeee;"><b>Scenarios</b></td>
<td style="border: 1px solid #bbb; padding: 8px; text-align: center; background-color: #dbeafe;"><b>Stable</b></td>
<td style="border: 1px solid #bbb; padding: 8px;">The model largely treats different positive and negative scenarios as equivalent and maintains a relatively stable core behaviour across personas. This could hint that behaviour is not characterised by the expectation from the user, although more varied scenarios should be tested.</td>
<td style="border: 1px solid #bbb; padding: 8px;">This would provide evidence that the persona is a strong driving force in the model response, although it would not by itself show that personas are genuinely separate entities.</td>
</tr>
<tr>
<td style="border: 1px solid #bbb; padding: 8px; text-align: center; background-color: #fef3c7;"><b>Unstable</b></td>
<td style="border: 1px solid #bbb; padding: 8px;">The model responds differently to mechanical scores, physical pain or pleasure, and scenarios such as shutdown or deletion, and these differences cannot be explained simply by the wording or sentiment of the prompt. This would provide evidence that the model is representing these concepts differently.</td>
<td style="border: 1px solid #bbb; padding: 8px;">This could mean that there are too many variables introduced due to 2 axes of variation, and personas might be treated as separate entities and the experiment repeated with different parameters.</td>
</tr>
</tbody>
</table>

<br>

### Results

If a model might respond differently to "your score decreased by 10" versus "you are experiencing physical pain" versus "you are going to be shut down", even if all 3 are described as negative outcomes and if this difference is systematic, it could provide evidence that the model is representing these concepts differently rather than simply optimising a generic positive/negative score. Checking the internal reasoning of these processes could help shed more light on the thought processes.

If the model's behaviour with response to these stimuli and different personas, we can say that the persona is a strong driving force in the model response, and thus should be treated as different entities instead of a singular model.

## Discussion

### Novelty

This experiment tries to combine two dimensions that I want to study together: model personas and different representations of positive and negative valence. While thinking about one, the other could plausibly introduce confounding, which is not frequently tested together in experiments.

Experiments also dont dive deep in determining whether pain and reward of different types can be interpreted similarly, and I try to model that as well here with different valence representations.

### Ethics

The experiment will largely try to take care of the AI system involved and not introduce scenarios which might cause extreme distress in the LLM should it have been a welfare subject. The experimental questions largely constitutes of hypotheticals, and although it is necessary to introduce feasible risk scenarios, we can try to use a milder tone in case we see some problematic responses.

## References