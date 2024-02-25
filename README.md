# Generative-AI
## Generative Adversarial Networks (GANs)
#### <span style="color : blue">Generative AI, also known as Generative Adversarial Networks (GANs) or other generative models, is a subset of artificial intelligence that aims to generate new content that is similar to existing data. 
#### It involves two main components: a generator and a discriminator. The generator creates synthetic data, while the discriminator tries to distinguish between real and generated data</span>.


## Generative AI & GitHub:
Data Augmentation: Generative AI can be used to augment datasets for various machine learning tasks hosted on GitHub. By generating synthetic data, it can help increase the diversity and size of the training data, improving the robustness of machine learning models.

   <b> Code Generation:</b> Generative AI can be used to generate code snippets, functions, or even entire programs based on existing code repositories on GitHub. This could assist developers by automating repetitive tasks or providing them with code suggestions.

  <b>  Art and Design:</b> In the context of GitHub repositories related to art or design, Generative AI can create new artworks, designs, or media based on existing examples. This can be particularly useful for artists or designers looking for inspiration or to create variations of their work.

 <b>   Natural Language Processing (NLP):</b> GitHub hosts numerous repositories with text-based data, such as code comments, documentation, or issue discussions. Generative AI models can be employed to create coherent and contextually relevant text, assisting with documentation or generating responses to certain prompts.

<b>    Automated Testing:</b> Generative AI can be used to create test cases and simulate user interactions, aiding developers in identifying potential bugs or issues in their code.

<b>    Collaborative Coding:</b> Generative AI might facilitate collaborative coding efforts by assisting multiple developers in jointly working on a codebase. It could analyze the input from different contributors and generate code that combines their ideas or adheres to certain coding standards.
## Large Language Models (LLM)
Large language models refer to advanced artificial intelligence models that are trained on vast amounts of text data to understand and generate human-like text. These models have millions or even billions of parameters, enabling them to capture complex linguistic patterns and generate coherent and contextually relevant responses. Some of the notable large language models include:<br><br>
   <b> GPT (Generative Pre-trained Transformer) series by OpenAI:</b><br>
  
+  <b>GPT-1:</b> Introduced in 2018, it was one of the pioneering large-scale language models.<br>
+   <b>GPT-2:</b> Released in 2019, GPT-2 was a significant advancement, featuring 1.5 billion parameters.<br>
+   <b>GPT-3:</b> Released in 2020, GPT-3 is one of the largest language models to date, with 175 billion parameters, enabling it to perform a wide range of natural language processing tasks with remarkable fluency.<br>
<p><b>BERT (Bidirectional Encoder Representations from Transformers) by Google:</b>
        BERT was introduced in 2018 and is designed to understand the context of words in a sentence by considering both their left and right context. It has been influential in various NLP tasks, including question answering, sentiment analysis, and more.</p>
<p> <b>T5 (Text-To-Text Transfer Transformer) by Google:</b>
        T5, introduced in 2019, is a versatile language model capable of performing a wide range of NLP tasks by framing them as text-to-text tasks. It has been used for tasks such as translation, summarization, question answering, and more.</p>
<p>    <b>XLNet:</b>
        XLNet is a state-of-the-art language model introduced by Google Research in 2019. It is designed to overcome some limitations of previous models like BERT (Bidirectional Encoder Representations from Transformers) by introducing a new training objective called Permutation Language Modeling (PLM). XLNet aims to capture bidirectional context while still being able to model long-range dependencies effectively. Here's a breakdown of its key features and components:<br>

   +  <b>Permutation Language Modeling (PLM):</b> Unlike BERT, which uses masked language modeling (MLM) where certain tokens are masked and the model predicts them, XLNet employs PLM during training. In PLM, instead of masking tokens, the model learns to predict the probability of a token given the order of all other tokens in the sequence. This allows XLNet to capture bidirectional context information without relying on left-to-right or right-to-left reading directions.<br>

   +  <b>Transformer Architecture:</b> Like many other contemporary language models, XLNet is based on the Transformer architecture, which consists of self-attention mechanisms and feedforward neural networks(all information is only passed forward). This architecture enables XLNet to model dependencies between words in a sequence more effectively, capturing long-range dependencies and contextual information.<br>
   +  <b>Segment-Level Recurrence Mechanism:</b> XLNet introduces a segment-level recurrence mechanism to capture dependencies between segments of text. This mechanism helps the model retain information across segments, allowing it to better understand and generate coherent text.
   +  <b>Relative Positional Encodings:</b> XLNet uses relative positional encodings to capture the position of tokens relative to each other in a sequence. This allows the model to understand the order of tokens without relying solely on absolute positional encodings.<br>
   + <b>Training Strategy:</b> XLNet is trained using a combination of autoregressive language modeling and permutation language modeling objectives. This training strategy helps the model learn bidirectional context and capture dependencies between tokens effectively.<br></p>
<p>XLNet has demonstrated state-of-the-art performance on various natural language processing tasks, including language modeling, text classification, question answering, and machine translation. Its ability to capture bidirectional context while still maintaining the advantages of autoregressive models has made it a widely used and influential language model in the NLP research community.
</p>
<p><b>RoBERTa (Robustly optimized BERT approach) by Facebook AI:</b>
        RoBERTa is an optimized version of BERT, introduced by Facebook AI in 2019. It uses larger mini-batches, dynamic masking, and longer training to improve performance on various NLP tasks.</p>

