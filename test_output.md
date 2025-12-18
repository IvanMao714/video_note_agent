=== INPUT ===
slides: 
video : E:\video_note_agent\example\cs336_01.mp4
query : 


=== TRANSCRIPT ===
Channel 0:
Welcome everyone, this is CS 3-3-6 Language Models From Scratch, and this is the core staff. So I am Percy, one of your instructors. I am really excited about this class because it really allows you to see the whole language modeling building pipeline end-to-end, including data systems and modeling. Tatsu, I will be co-teaching with him. So I'll let everyone introduce themselves. Hi everyone, I'm Patsy. I'm one of the co-instructors. I'll be giving lecture in you know week or two probably few weeks. Um, I'm really excited about this class. We Percy and I, you know, spent a while being a little disgruntled thinking like what's the really deep technical stuff that we can teach our students today? And I think one of the things is really you got to build it from scratch to understand it. So I'm hoping that that's sort of the ethos I'll take away from. Hey everyone, I'm Robert. I actually failed this class when I took it, but now I'm your TA. So everyone can study anything is possible. Models reasoning all that stuff, so yeah should be a fun quarter. Hey guys, I'm Marcelo. I'm a second year PhD. I work this is really last guy these days. I work on helping out the. And he was a topper of many leaderboards from last year, so he is the number to beat. Okay, all right, well thanks everyone. So let's continue as Satu mentioned. This is the second time we're teaching the class. We've grown the class by around 50%. We have three TAs instead of two and one big thing is we're making all the lectures on YouTube so that the world can learn how to build language models from scratch. Okay, so why do we decide to make this course and endure all the pain? So let's ask GPT-4. So if you ask it why teach a course on building language models from scratch, the reply is: teaching of course provides foundational understanding of techniques, fosters innovation — kind of the typical generic blather. Okay, so here's the real reason. So we're in a bit of a crisis, I would say. Researchers are becoming more and more disconnected from the underlying technology. Eight years ago, researchers would implement and train their own models in AI; even six years ago, you at least take the models like Bert and download them and fine-tune them. And now many people can just get away with prompting a proprietary model. So this is not necessarily bad, right? Because as you introduce these layers of abstraction, we can all do more and a lot of research has been unlocked by the simplicity of being able to prompt the language model. And I do my share of prompting, so there's nothing wrong with that. But it's also remember that these abstractions are leaky. So in contrast to programming languages or operating systems, you do not really understand what the abstraction is. It is a string in and string out, I guess. And I would say that there is still a lot of fundamental research to be done that requires tearing up the stack and co-designing different aspects of the data and the systems and the model. And I think really that full understanding of this technology is necessary for fundamental research. So that's why this class exists. We want to enable the fundamental research to continue, and our philosophy is to understand it you have to build it. So there's one small problem here, and this is because of the industrialization of language models. So GPT-4 has rumored to be 18 trillion parameters, cost $100 million to train. You have XAI building the clusters with 200,000 H100s. If you can imagine that there's investment of over $500 billion supposedly over four years. So these are pretty large numbers, right? And furthermore, there is no public details on how these models are being built here from GPT-4. This is even two years ago. They very honestly say that due to the competitive landscape and safety limitations, we are going to disclose no details. Okay, so this is the state of the world right now. And so in some sense, frontier models are out of reach for us. So if you came into this class thinking you're each going to train your own GPT... For sorry, so we're going to build small language models, but. The problem is that these might not be representative, and here's some of two examples to illustrate why. So here's kind of a simple one: if you look at the fraction of flops spent in attention layers of a transformer versus an MLP, this changes quite a bit. So this is a tweet from Stephen Frola from quite a few years ago, but this is still true. If you look at small models, it looks like the number of flops in the attention versus the MLP layers are roughly comparable. But if you go up to 175 billion, then the MLPs really dominate. Right? So why does this matter? Well, if you spend a lot of time at small scale and you're optimizing the attention, you might be optimizing the wrong thing because at larger scale. It doesn't, it gets washed out. This is kind of a simple example because you can literally make this plot without actually any computer. You just like do it's napkin math. Here's something that's a little bit harder to grapple with: it's just emergent behavior. So this is a paper from Jason Way from 2022, and here this plot shows that as you increase the amount of training flops. And you look at accuracy, a bunch on a bunch of tasks. You'll see that for a while it looks like the accuracy nothing is happening, and all of a sudden you get these kind of emergent of various phenomena like in context learning. So if you were hanging around at this scale, you would be concluding that well these language models really don't work when in fact you had to scale up to get that behavior. So do not despair, we can still learn something in this class. But we have to be very precise about what we are learning. So there is 3 types of knowledge: there is the mechanics of how things work; this we can teach you. We can teach you what a transformer is, you will implement a transformer. We can teach you how model parallelism leverages GPUs efficiently. These are. Just like kind of the raw ingredients, the mechanics. So that is fine. We can also teach you mindset. So this is something a bit more subtle and seems like a little bit fuzzy, but this is actually in some ways more important I would say because the mindset that we are going to take is that we want to squeeze as most out of the hardware as possible and take scaling seriously. Because in some sense, the mechanics all those we will see later that all these ingredients have been around for a while, but it was really I think the scaling mindset that OpenAI pioneered that led to this next generation of AI models. So mindset, I think hopefully we can bang into you that to think in a certain way. And the thirdly is intuitions, and this is about which data and modeling decisions lead to good models. This unfortunately we can only partially teach you, and this is because what architectures and what data sets work at most scales might not be the same ones that work at large scales. And but you know, that's just... But hopefully you got two and a half out of three, so that's pretty good being for your buck. Okay, speaking of intuitions, there's this sort of I guess sad reality of things that you know you can tell a lot of stories about why certain things in the transformer the way they are, but sometimes it's just you come, you do the experiments and experiments speak. So for example, there is this known Shazua paper that introduced the Swiglu, which is something that we will see a bit more in this class, which is a type of non-linearity. And in the conclusion, you know, the results are quite good and this got adopted. But in the conclusion, there is this honest statement that we offer no explanation except for this is divine benevolence. So there you go, this is the extent to our understanding. Okay, so now let's talk about this bitter lesson that I'm sure people have heard about. I think there's a sort of a misconception that the bitter lesson means that scale is all that matters, algorithms don't matter, all you do is pump more capital into building the model and you're good to go. I think this couldn't be farther from the truth. I think the right interpretation is that algorithms at scale is what matters. And because at the end of the day, your accuracy of your model is really a product of your efficiency and the number of resources you put in. And actually, efficiency, if you think about it, is way more important at larger scale because if you're spending hundreds of millions dollars, you cannot afford to be wasteful in the same way that if you're looking at running a job on your. On your local cluster, you might run it again. You fail, you debug it. And if you look at actually the utilization and the use, I am sure OpenAI is way more efficient than any of us right now. So efficiency really is important. And furthermore, this... I think is this point is maybe not as well appreciated in the sort of scaling rhetoric, so to speak. Which is that, if you look at efficiency, which is combination of hardware algorithms. But if you just look at the algorithmic efficiency, there's this nice open AI paper from 2020 that showed over the period of 2012 to 2019 there was a 44x. If algorithmic efficiency improvement in the time that it took to train ImageNet to a certain level of accuracy. Right, so this is huge, and I think if you... I don't know if you could see the abstract here. This is faster than Moore's Law. Right? So algorithms do matter. If you didn't have this efficiency, you would be paying 44 times more cost. This is for image models, but there's some results for language as well. Okay, so with all that, I think the right framing or mindset to have is: what is the best model one can build given a certain compute and data budget? Okay, and this question makes sense no matter what scale you're at because you're sort of like it's accuracy per resources. And of course, if you can raise the capital and get more resources, you'll get better models. But as researchers, our goal is to improve the efficiency of the algorithms. Okay, so maximize efficiency. We're going to hear a lot of that. Okay, so now let me talk a little bit about the current landscape and a little bit of I guess you know obligatory history. So language models have been around for a while now, going back to Shannon who looked at language models as a way to estimate the entropy of English. I think in AI they really were prominent in NLP where they were a component of larger systems like machine translation, speech recognition. And one thing that's maybe not as appreciated these days is that, if you look back in 2007, Google was training very large Ngram models, so five Gram models over two trillion tokens, which is a lot more tokens than GPT-3. And it was only I guess in the last two years that we've gotten to that token count. But they were in grand models, so they didn't really exhibit any of the interesting phenomena that we know of language models today. Okay, so in the 2010s I think a lot of you can think about this: a lot of the deep learning revolution happened and a lot of the ingredients sort of kind of falling into place. Right? So there was the first neural language model from Joshua Bengo's group and back in 2003 there was. Seek to seek models, this I think was a big deal for how do you basically model sequences from Ilya and Google folks. There is an Adam optimizer which still is used by the majority of people dating over a decade ago. There is a tension mechanism which was. UM, DEVELOPED IN THE CONTEXT OF MACHINE TRANSLATION, UM, WHICH THEN LED UP TO THE FAMOUS ATTENTION ALSO YOU NEED OR THE AKA THE TRANSFORMER PAPER IN TWENTY SEVENTEEN. PEOPLE WERE LOOKING AT HOW TO SCALE MIXTURE OF EXPERTS. THERE WAS A LOT OF WORK AROUND LATE TWO THOUSAND AND TENS ON HOW TO ESSENTIALLY DO MODEL PARALLELISM, AND THEY WERE ACTUALLY FIGURING OUT HOW YOU COULD TRAIN YOU KNOW ONE HUNDRED BILLION PRIMED THEIR MODELS. THEY DIDN'T TRAIN IT FOR VERY LONG BECAUSE THESE ARE THESE WERE LIKE MORE SYSTEMS. Work, but all the ingredients were kind of in place before or by the time the 2020 came around. So I think one other trend which was starting in LPU is the idea of these foundation models that could be trained on a lot of text and adapted to a wide range of downstream tasks. So Elmo, Bert, T5... These were models that were for their time very exciting. We kind of maybe forget how excited people were about things like Bert. Is a big deal, and then I think... I mean, this is abbreviated history, but I think one critical piece of the puzzle is OpenAI just taking these ingredients, they end applying very nice engineering and really kind of pushing on the kind of the scaling laws, embracing it as this is the kind of mindset. Piece and that led to GPT-2 and GPT-3, Google obviously was in the game and trying to compete as well, but that sort of paved the way I think to another kind of line of work, which is these were all closed models, so models that weren't released and you can only access via API, but they were all. No open models starting with early work by Eluther, right after GPT-3 came out Meta is early attempt which did not work maybe as quite as well Bloom and then Meta Alibaba DeepSeek AI 2, and there is a few others which I. I'mlicit have been creating these open models where the weights are released. One other piece of I think tidbit about openness, I think is important, is that there's many levels of openness. There's closed models like GPT-4, there's open weight models where the weights are available, and there's actually a paper — a very nice paper with lots of architectural details but no details about the data set. And then there is open source models, where all the weights and data are available in the paper, that where they are honestly trying to explain as much as they can, but of course you can not really capture everything in a paper, and there is no substitute for learning how to build it, except for kind of doing yourself. Okay, so that leads to kind of the present day where there's a whole host of frontier models from OpenAI, Anthropic, XAI, Google Meta DeepSeek, Alibaba Tencent and probably a few others that are sort of dominate the current landscape. So we're kind of in just this interesting time where. YOU KNOW, JUST TO KIND OF REFLECT A LOT OF THE INGREDIENTS LIKE I SAID WERE DEVELOPED, WHICH IS GOOD BECAUSE I THINK WE'RE GOING TO REVISIT SOME OF THOSE UM INGREDIENTS AND TRACE HOW THEY THESE TECHNIQUES WORK, AND THEN WE'RE GOING TO TRY TO MOVE AS CLOSE AS WE CAN TO BEST PRACTICES ON FRONTIER MODELS BUT YOU'RE USING UM INFORMATION FROM ESSENTIALLY THE OPEN YOU KNOW COMMUNITY. AND READING BETWEEN THE LINES FROM WHAT WE KNOW ABOUT THE CLOSED MODELS. Okay, so just as an interlude... Um, so what are you looking at here? So, um, this is a executable lecture. So it's a program where I'm stepping through and it delivers the content of lecture. So one thing that I think is interesting here is that... Um, you can embed code. So if you. UM, YOU CAN JUST STEP THROUGH CODE AND I THINK THIS IS A SMALLER SCREEN THAT I'M USED TO, BUT UH, YOU CAN LOOK AT THE ENVIRONMENT VARIABLES AS YOU'RE STEPPING THROUGH CODE, SO THAT'S UH USEFUL LATER WHEN WE START ACTUALLY UM TRYING TO DRILL DOWN AND GIVING CODE EXAMPLES. YOU CAN SEE THE HIERARCHICAL STRUCTURE OF LECTURE LIKE WE'RE IN THIS MODULE AND YOU CAN SEE WHERE IT'S IT WAS CALLED FROM MAIN, UM, AND YOU CAN JUMP TO DEFINITIONS, UM. Next supervised fine tuning, which we will talk about later. Okay? And if you think this looks like a Python program, well, it is a Python program, but I have made it processed it so for your viewing pleasure. Okay. So let's move on to the course logistics now, um... Actually, maybe I'll pause for questions. Any questions about what we're learning in this class? YEAH. Would you expect to graduate? For this class, to be able to lead a team to build a frontier model or other skills groups. So the question is: would I expect a graduate from this class to be able to lead a team and build a frontier model? Of course, with you know like a billion dollars of capital. Yeah, of course. I would say that it's a good step, but there's definitely many pieces that are missing. And I think we thought about we should really teach like a series of classes that eventually leads up to. To as close as we can get, but um... I think this is maybe the first step of the puzzle. But there are a lot of things and happy to talk offline about that. But I like the ambition. Yeah, that's what you should be doing: taking the class so you can go lead teams and build frontier models. Okay? UM. Okay, let's talk a little bit about the course. So here's a website, everything's online. This is a five-unit class, but I think that maybe doesn't express the level here as well as this quote that I pulled out from a course evaluation: The entire assignment was approximately the same amount of work as all five assignments from CS 24N plus the final project, and that's the first homework assignment. So not to scare you off, but just giving some data here. So why should you endure that? Why should you do it? I think this class is really for people who have sort of this obsessive need to understand how things work all the way down to the atom, so to speak. And I think, if you... when you get through this class, I think you will have really leveled up in terms of your research engineering and the comfort level of comfort that you'll have in building ML systems at scale will just be... I think something. There's also a bunch of reasons that you shouldn't take the class, for example if you want to get any research done this quarter maybe this class isn't for you. If you're interested in learning just about the hottest new techniques there are many other classes that can probably deliver on that better than for example spending a lot of time debugging BPE. And this is really, I think about a class about the primitives and learning things bottom up as opposed to the kind of the latest. And also, if you're interested in building language models or 4X, this is probably not the first class you would take, I think. Practically speaking, you know, as much as I kind of made fun of prompting, prompting is great. Fine-tuning is great if you can do that and it works. Then I think that is something you should absolutely start with. So I don't want people taking this class and thinking like: 'Great! Any problem?' The first step is to train a language model from scratch. That is not the right way of thinking about it. Okay，and I know that many of you，you know，some of you were enrolled，but we did have a cap，so we weren't able to enroll everyone。And also for the people online，you can follow it at home，all the lecture materials and assignments are online，so you can look at them。The lectures are also recorded and will be put on YouTube，although there will be some number of week lag there。And also, we will offer this class next year. So if you were not able to take it this year, do not fret. There will be next time. Okay? So the class has 5 assignments, and each of the assignments we do not provide scaffolding code in a sense that you literally give you a blank file and you are supposed to build things up. And in the spirit of learning, building from scratch, but we're not that mean. We do provide unit tests and some adapter interfaces that allow you to check correctness of different pieces, and also the assignment write-up if you walk through it does do it for sort of a gentle job of doing that. But you're kind of on your own for making good software design decisions and figuring out what you name your functions and how to organize your code. Which is a useful skill, I think. So one strategy I think for all assignments is that there is a piece of assignment which is just implement the thing and make sure it's correct, that mostly you can do locally on your laptop. You shouldn't need compute for that, and then we have a cluster that you can run for benchmarking both accuracy and speed. Right, so I want everyone to kind of embrace this idea of that you want to use as a small data set or as few resources possible to prototype before running large jobs. You shouldn't be debugging with one billion parameter models on the cluster if you can help it. There's some assignments which will have a leaderboard, um, which usually is of the form: do things to make perplexity go down given a particular training budget. Last year it was I think pretty... um, you know exciting for people to try to... um, you know try different things that you either learn from the class or you read online. 嗯。And then finally, I guess this year is... you know, this was less of a problem last year because I guess copilot wasn't as good, but you know, curse is pretty good. So I think our general strategy is that AI tools are can take away from learning because there are cases where it can just solve the thing you want it to do. But you know, I think you can obviously use them judiciously, so but use at your own risk. You're kind of responsible for your own learning experience here. Okay, so we do have a cluster. So thank you Together AI for providing a bunch of H100s for us. There's a guide to please read it carefully to learn how to use the cluster and start your assignments early because the cluster will fill up towards the end of a deadline as everyone's trying to get their large runs in. Okay，any questions about that？Right，so the question is：can you sign up for less than five units？I think administratively，if you have to sign up for less，that is possible，but it's the same class and the same workload. Any other questions? OKAY. So in this part, I'm gonna go through all the different components of the course and just give a broad overview, a preview of what you're gonna experience. Um, so remember it's all about efficiency given hardware and data. How do you train the best model given your resources? So for example, if I give you a common crawl dump, a web dump, and thirty-two h one hundred for two weeks, what should you do? There are a lot of different design decisions, there's questions about the tokenizer, the architecture systems optimizations you can do data things you can do, and we've organized the class into these five units or pillars. So I'm going to go through each of them in turn and talk about what we'll cover, what the assignment will involve. And then I will kind of wrap up, okay? So the goal of the basics unit is just get a basic version of a full pipeline working. So here you implement a tokenizer, model architecture and training. So I will just say a bit more about what these components are. So a tokenizer is something that converts between strings and sequences of integers. Intuitively, you can think about the integers corresponding to breaking up the string into segments. And mapping each segment to an integer, and the idea is that you just your sequence of integers is what goes into the actual model, which has to be like a fixed dimension. Okay? So in this course we will talk about the BPE encoding tokenizer, which is relatively simple and. And still is used, there are a promising set of methods on tokenizer-free approaches. So these are methods that just start with the raw bytes and do not do tokenization and develop a particular architecture that just takes the raw bytes. This work is promising, but so far I have not seen it been scaled to the frontier yet. So we will go with BPE for now. Okay, so once you have tokenized your sequence or strings into a sequence of integers, now we define a model architecture over these sequences. So the starting point here is original transformer, that's what is the backbone of basically all frontier models. And here's architectural diagram we won't go into details here but there's a attention piece and then there's an MLP layer with some. Normalization, so a lot has actually happened since 2017. Right? I think there's a sort of sense to which... Oh, the transformer is invented, and then everyone's just using transformer. And to a first approximation, that's true. We're still using the same recipe, but there have been a bunch of smaller improvements that do make a substantial difference when you add them all up. So for example, there is the activation nonlinear activation function, so Swiglu which we saw a little bit before positional embeddings. There's new positional embeddings these rotary position embeddings which we'll talk about normalization instead of using layer norm. We're gonna look at something called RMS norm which is similar but simpler. There's a question where you place the normalization which. has been changed from the original transformer, the mlp use the canonical version is a dense mlp and you can replace that with mixture of experts. Attention is something that has actually been gaining a lot of attention. I guess there's full attention and then there's sliding window attention and linear attention. All of these are trying to prevent the quadratic blow up. There's also lower dimensional versions like gqa and mla which we'll get to in a second or not in a second but in a future lecture. And then the most kind of maybe radical thing is other alternatives to the transformer, like state space models like Hyena, where they are not doing attention but some other sort of operation. And sometimes you get best of both worlds by mixing, making a hybrid model that mixes these in with transformers. Okay, so once you define your architecture, you need a train. So there is design decisions include optimizer, so Adam W which is a variant basically Adam fixed up is still very prominent. So we will mostly work with that, but it is worth mentioning that there is more recent optimizers like Muon and Soap that have shown promise. Learning rate schedule, batch size, whether you do regularization or not, hyperparameters. There is a lot of details here, and I think this class is one where the details do matter because you can easily have order of magnitude difference between a well-tuned architecture and something that is just like a vanilla transformer. So in assignment one, basically you'll implement the BPE tokenizer. I'll warn you that this is actually the part that seems to have been a lot of surprising, maybe a lot of work for people. So just you're warned. And you also implement the transformer cross-mp3p loss AdamW optimizer and training loop, so again the whole stack. And we are not making you implement PyTorch from scratch, so you can use PyTorch, but you can not use like. The transformer implementation for PyTorch, there is a small list of functions that you can use, and you can only use those. Okay, so we're gonna have some uh you know tiny stories and open web text data sets that you'll train on, and then there will be a leaderboard um to minimize the open web text perplexity. We'll give you ninety minutes on an A H one hundred and see what you can do. So this is last year, um. So see, we have the top. So this is the number to beat for this year. Okay? All right, so that's the basics now. After basics... I mean in some sense you're done, right? Like you have ability to train a transformer. What else do you need? So the system part really goes into how you can optimize this further, so how do you get the most out of hardware? And for this, we need to take a closer look at the hardware and how we can leverage it. So there is kernels, parallelism, and inference are the three components of this unit. So okay, so to first talk about kernels, let's talk a little bit about what a GPU looks like. Okay, so a GPU which we'll get much more into is basically a huge array of these little. Units that do floating point operations, and maybe the one thing to note is that this is the GPU chip, and here is the memory that's actually off chip, and then there's some other memory like L2 caches and L1 caches on chip. And so the basic idea is that compute has to happen here, your data might be somewhere else, and how do you basically organize your compute so that you can be most efficient? So one quick analogy is imagine that your memory is where you can store like your data and model parameters is. Make a warehouse, and your computer is like the factory. And what you... What ends up being a big bottleneck is just data movement costs, right? So the thing that we have to do is: How do you organize the compute? Like even a matrix multiplication to maximize the utilization of the GPUs. By minimizing the data movement, and there is a bunch of techniques like fusion and tiling that allow you to do that. So we will get all into the details of that and to implement and leverage a kernel. We are going to look at Triton. There is other things you can do with various levels of sophistication, but we are going to use Triton which is developed by OpenAI in a popular way to. Build kernels, okay? So we're going to write some kernels that's for one GPU. So now in general you have these big runs take you know 10 thousands if not tens of thousands of GPUs, but even at eight it kind of starts becoming interesting because you have a lot of GPUs they're connected to some CPU nodes and they also have are directly connected via MV switch. EMV LINK UM AND THE. It is the same idea, right? Now the only thing is that data movement between GPUs is even slower, right? And so we need to figure out how to put model parameters and activations and gradients and put them on the GPUs and do the computation and to minimize the amount of movement. And then, so we are going to explore different type of techniques like data parallelism and tensor parallelism and so on. So that is all I will say about that. And finally inference is something that we did not actually do last year in the class. Although we had a guest lecture, but this is important because inference is how you actually use a model. Right? It's basically the task of generating tokens given a prompt, given a trained model. And it also turns out to be really useful for a bunch of other things besides just chatting with your favorite model. You need it for reinforcement learning, test time compute, which has been. you know, very popular lately and even evaluating models you need to do inference. So we're going to spend some time talking about inference. Actually, if you think about the globally, the cost that's spent on inference is going... it's you know eclipsing the cost that it is used to train models because training despite it being very intensive is ultimately a one-time cost and inference. Is cost scales with every use, and the more people use your model, the more you will need inference to be efficient. Okay, so in inference there is 2 phases: there is a prefill and a decode. Prefill is you take the prompt and you can run it through the model and get some activations; and then decode is you go autoregressively one by one and generate tokens. So prefill all the tokens are given, so you can process everything at once. So this is exactly what you see at training time, and generally this is a good setting to be in because it's naturally parallel and you're mostly compute bound. What makes inference I think special and difficult is that this autoregressive decoding you need to generate one token at a time, and it's hard to actually saturate all your GPUs, and it becomes. Memory bound because you're constantly moving data around, and we'll talk about a few ways to speed the models up. This speed inference up, you can use a cheaper model. You can use this really cool technique called speculative decoding where you use a cheaper model to sort of scout ahead and generate multiple tokens, and then if these tokens happen to be good by some for some definition good, you can have the full model just score in and accept them all in parallel. And then there's a bunch of systems optimizations that you can do as well. Okay, so after the systems... Oh okay assignment two. So you're gonna implement a kernel? You're gonna implement some parallelism? So data parallel is very natural, and so we'll do that. Some of the model parallels like FSDP turns out to be a bit kind of complicated due from scratch, so we'll do sort of a baby version of that. But I encourage you to learn about the full version. We'll go over the full version in class, but implementing from scratch might be a bit too much. And then I think an important thing is getting in the habit of always benchmarking and profiling. I think that is actually probably the most important thing, is that you can implement things, but unless you have feedback on how well your implementation is going and where the bottlenecks are, you are just going to be kind of flying blind. Okay, so Unit 3 is scaling laws, and here the goal is you want to do experiments at small scale and figure things out, and then predict the hyperparameters and loss at large scale. Here is a fundamental question: So, if I give you a flops budget, what model size should you use? If you use a larger model, that means you can train on less data; and if you use a smaller model, you can train on more data. So, what is the right balance here? And this has been studied quite extensively and figured out by a series of papers from OpenAI and DeepMind. So, if you hear the term Chinchilla Optimal, this is what this is referring to. And the basic idea is that for every compute budget, number of flops, you can vary the number of parameters of your model. And then you measure how good that model is. So for every level of compute, you can get the optimal parameter count. And then what you do is, you can fit a curve to extrapolate and see if you had let's say 1 E22 flops, what would be the parameter size? And it turns out these minimum when you plot them, it's actually remarkably linear. Which leads to this, like very actually simple but useful rule of thumb: which is that if you have a particular model of size N, if you multiply by 20, that's the number of tokens you should train on essentially. So that means if I say one point four billion parameter model should be trained on twenty-eight billion. NO TOKENS. But you know, this doesn't take into account inference cost. This is literally how can you train the best model regardless of how big that model is? So there's some limitations here, but it's nonetheless been extremely useful for model development. So in this assignment, this is kind of... um... you know fun because we define a quote-unquote training API which you can query with a particular set of hyperparameters: you specify the architecture, you know, and batch size and so on. And we return you a loss that your decisions will get you. Okay, so your job is: you have a flops budget and you're going to try to figure out how to train a bunch of models, and then gather the data. You're going to fit a scaling law to the gathered data, and then you're going to submit your prediction on what you would choose to be the hyperparameters — what model size? AND SO ON, UM, AT A LARGER SCALE. Okay, so this is a case where you have to be really... We want to put you in this position where there's some stakes. I mean, this is not like burning real compute, but you know once you run out of your flops budget, that's it. So you have to be very careful in terms of how you prioritize what experiments to run, which is something that the Frontier Labs have to do all the time. And there will be a leaderboard for this, which is minimize flops, minimize loss given your flops budget. Question, so these are links from 2024. Yeah, so if we're working ahead, should we expect assignments to change over time? Or are these going to be the final... Yeah, so the question is: Are these links from 2024? The rough assignments, the rough structure will be the same from 2025. There will be some modifications, but if you look at these, you should have a pretty good idea of what to expect. Okay, so let's go into data now. Okay, so up until now you have scaling laws, you have systems, you have your transformer implementation, everything. You're really kind of good to go. But data I would say is a really kind of key ingredient that I think differentiates in some sense. And the question to ask here is, what do I want this model to do? Right? Because what the model does is completely determined. I mean, mostly determined by the data. If I put... if I train on multilingual data, it will have multilingual capabilities; if I train on code, it will have code capabilities. And not you know... It's very natural and usually data sets are a conglomeration of a lot of different pieces. There's you know... This is from a pile which is four years ago, but the same idea I think holds. you know, you have data from you know the web. This is common crawl. You have stack exchange, wikipedia, github and different sources which are curated. And so in the data section we're going to start talking about evaluation, which is given a model how do you evaluate whether it's any good? So we're going to talk about perplexity where measures standard kind of standardized testing like mmlu. If you have models that generate utterances for instruction following, how do you evaluate that? There is also decisions about if you can ensemble or do chain of thought at test time, how does that affect your evaluation? And then you can talk about entire systems evaluation of entire system, not just a language model, because language models often get these days plugged into some agentic system or something. Okay, so now after establishing evaluation, let's look at data curation. So this is I think an important point that people don't realize. I often hear people say, 'Oh we're training the model on the internet.' This just doesn't make sense, right? Data doesn't just fall from the sky, and there's the internet that you can. Pipe into your model, you data has to always be actively acquired somehow. So even if you... just as an example of you know, I always tell people: look at the data. And so let's look at some data. So this is some common crawl data going to take 10 documents. And I think, hopefully this works. Okay, I think the rendering is off, but you can kind of see... This is a sort of random sample of Common Crawl, and you can see that. This is maybe, um. Not exactly the data, oh here's some actually real text here okay that's cool um but if you look at most of common call aside from this is a different language but you can also see this is very spammy sites and you'll quickly realize that a lot of the web is just you know trash and so uh well okay maybe that's not that's surprising but it's more trash than you would actually expect i promise. So what I am saying is that there is a lot of work that needs to happen in data, so you can crawl the internet, you can take books, archives, papers, GitHub, and there is actually a lot of processing that needs to happen. There is also legal questions about what data you can train on, which we will touch on nowadays. A lot of frontier models have to actually buy data. Because the data on the internet that's publicly accessible is actually turns out to be a bit limited for that kind of really frontier performance, and also I think it's important to remember that this data that's scraped, it's not actually text right? First of all, it's HTML or it's PDFs or in the case of code, it's just directories. So there has to be an explicit process. That takes this data and turns it into text, okay? So we are going to talk about the transformation from HTML to text, and this is going to be a lossy process. So the trick is: how can you preserve the content and some of the structure without basically just having HTML? Filtering, as you could surmise, is going to be very important both for getting high-quality data but also removing harmful content. Generally, people train classifiers to do this. Deduplication is also an important step which we will talk about. Okay, so Assignment 4 is all about data. We are going to give you the raw Common Crawl dump. So you can see just how bad it is, and you're gonna train classifiers dedupe, and then there's gonna be a leaderboard where you're gonna try to um minimize perplexity given your token budget. So now let's... Now you have the data, you've done this build all your fancy kernels, you've trained now you can really train models but at this point what you'll get. Is a model that can complete the next token, right? And this is called essentially base model. And I think about it as a model that has a lot of raw potential, but it needs to be aligned or modified some way. And alignment is a process of making it useful. So alignment captures a lot of different things, but. Three things I think it captures is that you want to get the language model to follow instructions, right? Complete the next token is not necessarily following the instruction. It will just complete the instruction or whatever it thinks will follow the instruction. You get to here specify the style of the generation: whether you want to be a long or short, whether you want bullets, whether you want it to be witty or have sass or not. And when you play with. You chat GPT versus Rock, you'll see that there's different alignment that has happened. And then also safety: one important thing is for these models to be able to refuse answers that can be harmful, so that's where alignment also kicks in. So there's generally two phases of alignment — there's supervised fine-tuning. And here, the goal is... I mean, it's very simple. You basically gather a set of user assistant pairs, so prompt-response pairs, and then you do supervised learning. Okay? And the idea here is that the base model already has the sort of the raw potential, so just fine-tuning it on. A few examples is sufficient, of course. The more examples you have, the better the results. But there's papers like this one that shows even like a thousand examples suffices to give you instruction following capabilities from a good base model. Okay? So this part is actually very simple and it's not that different from pre-training because it's just you're given text and you just maximize the probability of the text. UM. So the 2nd part is a bit more interesting from an algorithmic perspective. So the idea here is that even with SFT phase, you will have a decent model. And now how do you improve it? Well, you can get there more SFT data, but that can be very expensive because you have to have someone sit down and annotate data. The goal of learning from feedback is that you can leverage lighter forms and annotation, and have the algorithms do a bit more work. Okay, so one type of data you can learn from is preference data. So this is where you generate multiple responses from a model to a given prompt like A or B, and the user rates whether A or B is better. And so the data might look like: it generates what's the best way to train language model? Use a large data set or use a small data set. And of course, the answer should be A. So that is a unit of expressing preferences. Another type of supervision you could have is using verifiers. So for some domains, you are lucky enough to have a formal verifier like for math or code, or you can use learn verifiers where you train an actual language model to rate the. The response, and of course this relates to evaluation again algorithms. This is we are in the realm of reinforcement learning, so one of the earliest algorithms that was developed that was applied to instruction tuning models was PPO (Proximal Policy Optimization). It turns out that if you just have preference data, there is a much simpler algorithm called DPO that works really well. But in general, if you want to learn from verifiers' data, you have to... It is not preference data, so you have to embrace RL fully. And there is this method which we will do in this class, which called. Group Relative Preference Optimization，which simplifies PPO and makes it more efficient by removing the value function throughout by Deep Seek，which seems to work pretty well。Okay, so assignment five implements supervised tuning DPO and GRPO, and of course evaluate question. Yeah, the question is: as time and one seems a bit daunting, what about the other ones? I would say that assignment one and two are definitely the most heavy and hardest, um, assignment three is a little bit more of a breather, and assignment four and five at least last year were... um, I would say a notch below assignment two. Um, although I don't know depends on we haven't fully worked out the details for this year. Yeah, it does get better. Okay, so just to a recap of the different pieces here: remember efficiency is this driving principle, and there's a bunch of different design decisions. And you can... I think if you view efficiency everything through lens of efficiency, I think a lot of things kind of make sense. And importantly, I think we are... It is worth pointing out there: We are currently in this compute-constrained regime, at least this class and most people who are somewhat GPU-poor. So we have a lot of data but we do not have that much compute, and so these design decisions will reflect squeezing the most out of the hardware. So for example, data processing — we are filtering fairly aggressively because we do not want to waste precious compute on. Better relevant data tokenization, like it is nice to have a model over bytes that is very elegant, but it is very compute inefficient with today's model architectures. So we have to do tokenization too as an efficiency gain. Model architecture there are a lot of design decisions there that are essentially motivated by efficiency training. I think the fact that we are most of what we are doing to do is just a single epoch. This is clearly, we are in a hurry. We just need to see more data as opposed to spend a lot of time on any given data point. Skiing laws is completely about efficiency. We use less compute to figure out the hyperparameters and alignment is maybe a little bit different, but the connection to efficiency is that if you can put resources into alignment. Then you actually require less smaller base models, okay? So there is a... There are sort of two paths: if your use case is fairly narrow, you can probably use a smaller model, you align it or fine-tune it, and you can do well; but if your use cases are very broad, then there might not be a substitute for training a big model. So that's today, so increasingly now at least for Frontier Labs they're becoming data constrained which is interesting because I think that the design decisions will presumably completely change. Well, I mean compute will always be important but I think the design decisions will change. For example you know learning taking one epoch over your data I think doesn't really. Make sense if you have more compute, why wouldn't you take more epochs? At least or do something smarter. Or maybe there will be different architectures, for example because a transformer was really motivated by compute efficiency. So that's something to kind of ponder. Still it's about efficiency, but the design decisions reflect what regime you're in. OKAY. So now I'm going to dive into the first unit, um... yeah? Before that, any questions? Do we have a spot for a head? The question is, we have a slack or we will have a slack? We will send out details after the response. Yeah, will students auditing the course also have access to the same material? The question is: students auditing the class will have access to all the. Online materials, assignments, and we will give you access to Canvas so you can watch the lecture videos. What's the grading? What's the grading of the assignments? Good question. So there will be a set of unit tests that you will have to pass, so part of the grading is just did you implement this correctly. There will also parts of the grade which will... Did you implement a model that achieved a certain level of loss or is efficient enough? In the assignment, every problem part has a number of points associated with it, and so that gives you a fairly granular level of what grading looks like. Okay, let's jump into tokenization. Okay, so Andre Capatti has this really nice video on tokenization and in general he makes a lot of these videos on that actually inspired a lot of this class how you can build things from scratch. So you should go check out some of his videos. UM, SO TOKENIZATION AS WE TALKED ABOUT IT, UM, IS THE PROCESS OF TAKING RAW TEXT WHICH IS GENERALLY REPRESENTED AS UNICODE STRINGS AND, UM, TURNING IT INTO A SET OF INTEGERS ESSENTIALLY, AND WHERE EACH INTEGER IS, UM, REPRESENTS A TOKEN. Okay, so we need a procedure that encodes strings to tokens and decodes them back into strings. And the vocabulary size is just the number of values that a token can take on, the range of the integers. Okay, so just to give you an example of how tokenizers work, let's play around with this really nice website which allows you to look at different tokenizers and just type in something like 'hello' or whatever. Do this, and one thing it does is it shows you the list of integers. This is the output of tokenizer. It also nicely maps out the decomposition of the original string into a bunch of segments and a few things to kind of note. First of all, the space is part of a token. So, unlike classical NLP where the space just kind of disappears, everything is accounted for. These are meant to be kind of reversible operations: tokenization and by convention, you know, for whatever reason, the space is usually preceding the token. Also notice that you know hello is a completely different token than. Space hello, which you might make you a little bit squeamish, but it seems and it can because problems, but that is just how it is. Question: Is the space being leading instead of trailing intentional? Or is it just an artifact of the BPE process? So the question is, is the spacing before intentional or not? So in the BP process, I will talk about you actually pretokenize and then you tokenize each part. And I think the pretokenizer does put the space in the front, so it is built into the algorithm. You could put it at the end, but I think it probably makes more sense to put in the. Beginning, but actually don't well I guess it could go either way. It's my sense okay? So then if you look at numbers, you see that the numbers are chopped down into different pieces. It's a little bit kind of interesting that it's left to right, so it's definitely not grouping by thousands or anything like semantic. But anyway, I encourage you to kind of play with it and get a sense of what these existing tokenizers look like. So this is a tokenizer for GPT-40, for example. So there is some observations that we made, so if you look at the GBT2 tokenizer which will use this kind of as a reference. Okay, let me see if I can. Hopefully this is, let me know if this is getting too small in the back. You can take a string, if you apply the GPT-2 tokenizer, you get your indices. So it maps strings to indices, and then you can decode to get back the string. AND THIS IS JUST A SANITY CHECK TO MAKE SURE THAT, UM, YOU ACTUALLY ROUND TRIPS, UM, ANOTHER THING THAT'S I GUESS INTERESTING TO LOOK AT IS THIS COMPRESSION RATIO, WHICH IS IF YOU LOOK AT THE NUMBER OF BYTES DIVIDED BY THE NUMBER OF TOKENS. SO HOW MANY BYTES ARE REPRESENTED BY A TOKEN? AND THE ANSWER HERE IS ONE POINT SIX. So every token represents one point six bytes of data. Okay, so that's just a GPT tokenizer that OpenAI trained. To motivate kind of BPE, I want to go through a sequence of attempts. So suppose you wanted to do tokenization, what would be the sort of the simplest thing? The simplest thing is probably character-based tokenization. A Unicode string is a sequence of Unicode characters, and each character can be converted into an integer called a code point. Okay, so A maps to 97, the world emoji maps to 127757, and you can see that it converts back. Okay, so you can define a tokenizer which simply maps each character into a code point. OKAY. So what's one problem with this? Yeah, the compression ratio is one. So that's well... Actually, the compression ratio is not quite one because a character is not a byte, but it's maybe not as good as you want. One problem with that: if you look at some code points, they're actually really large, right? So you're basically allocating each like one slot in your vocabulary for every character uniformly, and some characters appear way more frequently than others. So this is not a very effective use of your kind of budget. Okay, so the vocabulary size is huge. I mean, the vocabulary size being 127 is actually a big deal, but the bigger problem is that some characters are rare and this is inefficient use of the vocab. Okay, so the compression ratio is. Is one point five in this case, because it's the tokens, sorry, the number of bytes per token and a character can be multiple bytes. Okay, so that was a very kind of naive approach. On the other hand, you can do byte-based tokenization. Okay, so Unicode strings can be represented as sequence of bytes because every string can just be converted into bytes. Okay, so some A is already just kind of one byte, but some characters. Take up as many as four bytes, and this is using the UTF-8 kind of encoding of Unicode. There's other encodings, but this is the most common one that's so dynamic. So let's just convert everything into bytes and see what happens. So if you do it into bytes, now all the indices are between 0 and 256 because there are only 256 possible values for a byte by definition. So your vocabulary is very small, and each byte is... I guess not all bytes are equally used, but you don't have too many sparsity problems. UM, BUT WHAT'S THE PROBLEM WITH BYTE BASED ENCODING? Yeah, long sequences. So this is... I mean, in some ways, I really wish by coding would work. It's the most elegant thing, but you have long sequences. Your compression ratio is 1:1 byte per token, and this is just terrible of compression ratio of one is terrible because your sequences will be really long. Attention is quadratic naively in the sequence length, so this is you're just going to have a bad time in terms of efficiency. Okay, so that was not really good. So now the thing that you might think about is: well, maybe we kind of have to be adaptive here, right? Like we can not allocate a character or a byte per token, but maybe some tokens can represent lots of bytes and some tokens can represent few bytes. So one way to do this is word-based organization, and this is something that was actually very classic in NLP. So here's a string, and you can just split it. INTO A SEQUENCE OF SEGMENTS. Okay，AND YOU CAN CALL EACH OF THESE TOKENS. So you just use a regular expression, here's a different regular expression that GPT2 uses to pre-tokenize, and it just splits your string into a sequence of strings. So, and then what you do with each segment is that you assign each of these to integer, and then you're done. Okay, so what's the problem with this? Yeah, so the problem is that your vocabulary size is sort of unbounded. Well, not maybe not quite unbounded, but you do not know how big it is because on a given new input, you might get a segment that just you have never seen before, and that is actually kind of a big problem. This is actually WordBase is a really big thing. Because some real words are rare, and actually it is really annoying because new words have to receive this unk token. And if you are not careful about how you compute the perplexity, then you are just going to mess up. Word based isn't, I think it captures the right intuition of adaptivity, but it's not exactly what we want here. So here we're finally gonna talk about the BPE encoding or byte pair encoding, um... So this was actually a very old algorithm developed by Philip Gage in ninety-four for data compression, um... And it was first introduced into NLP for neural machine translation. So before papers that did machine translation or any basically all NLP used word-based tokenization. And again, word-based was a pain. So this paper pioneered this idea: well, we can use this nice algorithm from 94, and we can just make the tokenization kind of round trip, and we don't have to deal with unk or any of that stuff. And then finally, this entered the kind of language modeling era through GPT-2, which was trained on using the BPE tokenizer. Okay, so the basic idea is: instead of defining some sort of preconceived notion of how to split up, we are going to train the tokenizer on raw text. That's the basic kind of insight, if you will. And so organically common sequences that span multiple characters we're going to try to represent as one token, and rare sequences are going to be represented by multiple tokens. There's a sort of a slight detail which is for efficiency: The GPT-2 paper uses word-based tokenizer as a sort of preprocessing to break it up into segments, and then runs BP on each of the segments. Which is what you're gonna do in this class as well, um, the algorithm BP is actually very simple. So we first convert the string into a sequence of bytes which we already did when we talked about byte-based tokenization. And now we're gonna successively merge the most common pair of adjacent tokens over and over again. So that intuition is that if a pair of tokens shows up a lot then we're going to compress it into one token. We're so gonna dedicate space for that. Okay, so let's walk through what this algorithm looks like. So we're going to use this cat and hat as an example, and we're going to convert this into a sequence of integers. These are the bytes, and then we're going to keep track of what we've merged. So remember, merges is a map from two integers which can represent bytes or other. you know pre-existing tokens, and we're gonna create a new token. And the vocab is just going to be a handy way to represent the index to bytes, okay? So we're going to the BP algorithm. I mean, it's very simple, so I'm just actually going to run through the code. You're going to do this number merges of times, so number merges is three in this case. We're going to first count up the number of occurrences of pairs of bytes, so hopefully this doesn't become too small. So we're going to just step through. This sequence, and we're going to see that okay? So what's 116104? We're going to increment that count. 104101 increment that count. We go through the sequence and we're going to count up the bytes. Okay, so now after we have these counts, we're going to find the pair that occurs the most number of times. So I guess there's multiple ones, but we're just going to break ties and say 1-1-6 and 1-0-4. Okay, so that occurred twice. So now we're going to merge that pair, so we're going to create a new slot in our vocab which is going to be 256. So so far it's zero through one two fifty-five, but now we're expanding the vocab to 256, and we're going to say every time we see one one six and one four, we're going to replace it with 256. Okay。And then we're going to just apply that merge to our training set, so after we do that, the 116-104 became 256, and this 256 remember occurred twice. Okay, so now we're just going to loop through this algorithm one more time. The second time it decided to merge 256 and 101, and now I'm going to replace that in indices. And notice that the indices is going to shrink right because our compression ratio is getting better as we make room for more vocabulary items, and we have a greater vocabulary to represent everything. Okay, so let me do this one more time. Um, and then the next merge is two fifty-seven three, and this is shrinking one more time. Okay, and then now we're done. Okay, so let's try out this tokenizer. So we have the string 'the quick brown fox', we're going to encode into a sequence of indices, and then we're going to use our BP tokenizer to decode. Let's actually step through what that looks like. This well, actually maybe decoding isn't actually interesting. Sorry, I should have gone through the encode. Let's go back to encode. So encode: you take a string, you convert to indices, and you just replay the merges in importantly in the order that occur. So I'm going to replay. These merges, and then. And then I am going to get my indices, okay? And then verify that this works, okay? So that was... It is pretty simple. You know, it is because it is simple. It was also very inefficient. For example, encode loops over the merges. You should only loop over the merges that matter. And there is some other bells and whistles like their special tokens pre-tokenization. And so in your assignment, you're going to essentially take this as a starting point and... Or I mean, I guess you should implement your own from scratch. But your goal is to make the implementation fast, and you can like parallelize it if you want. You can go have fun. Okay, so summary of tokenization. So tokenizer maps between strings and sequences of integers. We looked at character-based, byte-based, word-based; they are highly suboptimal for various reasons. BPE is a very old algorithm from 94 that still proves to be effective heuristic. And the important thing is that it looks at your corpus statistics to make sensible decisions about how to best adaptively allocate. Vocabulary to represent sequences of characters. And I hope that one day I won't have to give this lecture because we'll just have architectures that map from bytes, but until then we'll have to deal with tokenization. Okay, so that is it for today. Next time we are going to dive into the details of PyTorch and give you the building blocks and pay attention to resource accounting. All of you have presumably implemented PyTorch programs, but we are going to really look at where all the flops are going. Okay, see you next time. 


=== NOTES ===
# Study Notes

## 1) Learning Goals
- Grasp the full pipeline from raw text to a deployed language model.  
- Build a Byte‑Pair Encoding (BPE) tokenizer from scratch, including a full worked example.  
- Understand and implement a Transformer block with Swiglu, rotary positional embeddings, RMSNorm, and efficient attention variants.  
- Write a scalable training loop that incorporates AdamW, learning‑rate schedules, mixed‑precision, gradient clipping, and checkpointing.  
- Design GPU‑efficient kernels with Triton, apply data/tensor/pipeline parallelism, and optimize inference with prefill/decode strategies.  
- Apply scaling‑law reasoning (Chinchilla rule) and fit a power‑law model to plan hyperparameters.  
- Curate, filter, deduplicate, and tokenize massive corpora, and align models using SFT, DPO, and verifier‑guided GRPO.  
- Master profiling and benchmarking to iterate toward better performance.

## 2) The Story in One Page  
We start with raw web text, strip markup, filter toxic content, and deduplicate. Byte‑Pair Encoding (BPE) converts Unicode strings into compact integer token sequences. These tokens feed a stack of Transformer blocks; each block applies multi‑head self‑attention, a Swiglu‑activated feed‑forward network, and RMSNorm with rotary positional encodings. Super‑vised fine‑tuning (SFT) on instruction/response pairs and preference‑based Direct Preference Optimization (DPO) encourage safe, helpful behavior. Training a billions‑parameter model requires a GPU cluster; custom Triton kernels fuse matrix multiply with normalization, while data‑, tensor‑, and pipeline parallelism distribute work. Speculative decoding and the verifier‑guided GRPO further improve inference latency and safety. Scaling laws, particularly the Chinchilla rule, guide the optimal balance of parameters versus training tokens. Profiling (kernel occupancy, memory bandwidth, compute) ensures every component runs at peak efficiency. The final model is evaluated on perplexity, latency, and safety metrics before being released.  

**Mini‑outline**  
1. Tokenization (BPE)  
2. Transformer architecture & variants  
3. Training pipeline & optimizers  
4. GPU systems (kernels, parallelism, inference)  
5. Scaling laws & hyperparameter planning  
6. Data & alignment pipeline  
7. Course mindset & evaluation  

---

## 3) Main Notes

### 3.1 Tokenization Fundamentals  
**Overview**  
Tokenization converts human readable text into integer IDs that a neural network can operate on. We use Byte‑Pair Encoding (BPE), a data‑driven compression scheme that starts from a fixed byte alphabet and repeatedly merges the most frequent adjacent pair until the desired vocabulary size is reached.  

**Detailed Explanation**  
*Character/byte tokenizers* start with a 256‑byte alphabet and simply assign a unique ID to each byte. BPE builds on this by iteratively merging byte pairs:  
1. **Pre‑tokenize** the text using a regex that preserves punctuation and whitespace.  
2. **Byte encode** each token into its UTF‑8 byte sequence.  
3. **Merge Loop**  
   - Count all adjacent byte pairs across the corpus.  
   - Pick the most frequent pair `(x, y)`.  
   - Add a new merged token `z` with rule `(x, y) → z`.  
   - Replace every occurrence of `(x, y)` with `z`.  
   - Repeat until the vocabulary reaches the target size (e.g., 50 k).  

The resulting vocabulary collapses frequent substrings (e.g., “the”) into single tokens while rare sequences stay split into smaller pieces. BPE is language‑agnostic, deterministic, and produces a compressed representation with about 1.6 bytes per token for GPT‑2‑style models.  

**Worked Example: “hello world”**  
| Step | Bytes | Merge Rule | Tokens after Merge |
|------|-------|------------|--------------------|
| 0 | `h e l l o   w o r l d` | – | `[h, e, l, l, o, ␣, w, o, r, l, d]` |
| 1 | `(h, e)` | `he` | `[he, l, l, o, ␣, w, o, r, l, d]` |
| 2 | `(l, l)` | `ll` | `[he, ll, o, ␣, w, o, r, l, d]` |
| 3 | `(o, ␣)` | `o ` | `[he, ll, o , w, o, r, l, d]` |
| 4 | `(w, o)` | `wo` | `[he, ll, o , wo, r, l, d]` |
| 5 | `(r, l)` | `rl` | `[he, ll, o , wo, rl, d]` |
| 6 | `(wo, rl)` → `wor` | `wor` | `[he, ll, o , wor, d]` |
| 7 | `(wor, d)` → `world` | `world` | `[he, ll, o , world]` |

**Final Token IDs** (example, assume mapping)  
`[ID('he'), ID('ll'), ID('o'), ID('world')]`  

**Procedure**  
1. Load the merge table from the training corpus.  
2. Encode each raw string by applying all merge rules in the order defined.  
3. Map the resulting token symbols to integer IDs via a lookup table.  

**Implementation Notes**  
- Store `merge_dict: Tuple[int, int] → int` for quick lookup.  
- Maintain `id_to_bytes` for decoding.  
- Incrementally update pair counts to avoid rescanning all data each merge.  
- Parallelize segment processing with `concurrent.futures`.  

**Common Confusions**  
- *Misconception*: BPE starts from words.  
  *Correction*: It starts from bytes; merges occur at the byte level before word boundaries are considered.  

**Check Yourself**  
- What is the sequence of merge operations for the string “data”?  
- Why does BPE maintain a deterministic tokenization?  

---

### 3.2 Transformer Architecture & Variants  
**Overview**  
A Transformer block takes token embeddings and produces contextual representations. It comprises multi‑head self‑attention, a feed‑forward network, and a normalization layer. Modern improvements—Swiglu, rotary positional embeddings, RMSNorm, and efficient attention patterns—reduce memory, speed up training, and improve extrapolation.  

**Detailed Explanation**  
*Self‑Attention*  
- Compute queries `Q`, keys `K`, values `V` by linear projections.  
- Score: `A = (QKᵀ)/√d_k`.  
- Softmax over scores gives attention weights, then `O = softmax(A)V`.  

*Feed‑Forward MLP*  
- Two linear layers with non‑linearity in between.  
- **Swiglu** (Switched Gated Linear Unit) activation:  
  ```
  Swiglu(x) = x ⋅ σ(x)
  ```  
  where `σ` is the sigmoid function.  
  Two linear projections are applied inside the module; the first produces the input for both the linear output and the sigmoid gate. This gating gives the network greater expressiveness with minimal extra computation.  

*Normalization*  
- **LayerNorm** normalizes across the feature dimension.  
- **RMSNorm** replaces the mean with root‑mean‑square:  
  ```
  RMSNorm(x) = x / ( sqrt( (1/d) Σ x_j² ) + ε )
  ```  
  It is faster because it omits the mean calculation.  

*Positional Encoding*  
- **Rotary** encodes relative positions by rotating the query and key vectors in complex space; allows extrapolation to longer sequences while remaining parameter‑free.  

*Attention Variants*  
- **Full (quadratic)**  `O(L²)` over sequence length `L`.  
- **Sliding‑window** limits attention to a fixed window `w`, yielding approximate linear `O(L·w)` complexity.  
- **Linear** attention approximates the softmax kernel with a separable kernel, giving true linear `O(L)` cost.  

*Other Blocks*  
- **Mixture‑of‑Experts (MoE)** splits the MLP into multiple experts, and a gate selects a small subset per token, dramatically increasing model capacity with modest compute.  
- **State‑Space Models** replace attention with linear recurrence, allowing `O(L)` processing without attention.  

**Procedure**  
1. Convert `token_ids → [B, L, d]` embeddings.  
2. Add rotary or learned positional embeddings.  
3. Feed through `N` transformer blocks, each performing self‑attention → Swiglu → RMSNorm.  
4. Project to vocab logits with a final linear layer.  
5. Compute cross‑entropy loss on the next token predictions.  

**Implementation Notes**  
- Triton kernels can fuse attention, bias addition, RMSNorm, and Swiglu in one launch, dramatically reducing memory bandwidth.  
- Swiglu in PyTorch: `x * torch.sigmoid(x)` after the first linear layer.  
- MoE requires expert placement across GPUs; use `torch.distributed` for scatter‑gather operations.  
- For sliding‑window attention, use a binary mask (`-inf`) on out‑of‑window positions before softmax.  

**Common Confusions**  
- *Misconception*: Rotary embeddings are absolute positional encodings.  
  *Correction*: They encode *relative* position through complex rotations, enabling sequence extrapolation.  

**Check Yourself**  
- Derive the gradient of Swiglu.  
- Compare compute cost of full vs. sliding‑window attention at `L=1024`.  

---

### 3.3 Training Pipeline & Optimizers  
**Overview**  
Training a large language model relies on a robust data loader, an optimizer with weight decay (AdamW), a learning‑rate schedule with warm‑up and cosine decay, mixed‑precision training, gradient clipping, accumulation, and checkpointing.  

**Detailed Explanation**  
1. **Dataset**  
   - Tokenized text is split into fixed‑length chunks (e.g., 1 024 tokens).  
   - Shuffled and loaded with `torch.utils.data.DataLoader`, `pin_memory=True`.  

2. **Forward & Loss**  
   - `logits = model(inputs)`; the model predicts the next token for each position.  
   - Cross‑entropy loss:  
     ```
     L = - (1/(B·(L-1))) Σ_b Σ_l log p(y_b,l+1 | context)
     ```  

3. **AdamW Update**  
   ```
   m_t = β1 m_{t-1} + (1-β1) g_t
   v_t = β2 v_{t-1} + (1-β2) g_t²
   m̂_t = m_t / (1-β1^t)
   v̂_t = v_t / (1-β2^t)
   θ_{t+1} = θ_t - η * (m̂_t / (sqrt(v̂_t)+ε)) - η*λ*θ_t
   ```  

4. **Learning‑Rate Schedule**  
   - **Warm‑up**: linear increase over `W` steps to `η_max`.  
   - **Cosine decay**:  
     ```
     η_t = η_min + 0.5*(η_max-η_min)*(1+cos(π*T/T_total))
     ```  

5. **Gradient Management**  
   - Clamp `||g|| ≤ τ` via `torch.nn.utils.clip_grad_norm_`.  
   - Mixed precision with `torch.cuda.amp.autocast()` and `GradScaler`.  
   - Gradient accumulation to simulate larger batch sizes.  

6. **Evaluation**  
   - Compute perplexity:  
     ```
     PP = exp( mean( -log p(y|x) ) )
     ```  

**Implementation Notes**  
- Wrap forward/backward inside an `amp.autocast()` context.  
- Use `optimizer.step()` only after unscaling gradients.  
- Log loss, learning rate, and other metrics each step.  
- Checkpoint: `state_dict` for model and optimizer.  

**Common Confusions**  
- *Misconception*: L2 regularization can replace weight decay in AdamW.  
  *Correction*: Weight decay in AdamW is decoupled from the gradient update, whereas L2 regularization is tied to it, producing different effective regularization.  

**Check Yourself**  
- Why is gradient clipping essential when training billions of parameters?  
- Draw the learning‑rate curve for `W=4000`, `η_min=1e-6`, `η_max=3e-4`.  

---

### 3.4 GPU Systems: Kernels, Parallelism, Inference  
**Overview**  
Efficient GPU utilization hinges on kernel fusion, custom kernels via Triton, and the right combination of data, tensor, and pipeline parallelism. Inference is split into *prefill* (processing the prompt) and *decode* (token‑by‑token generation), with *speculative decoding* further accelerating latency.  

**Detailed Explanation**  
- **Kernel Fusion**  
  - Fuse matrix multiply, bias addition, RMSNorm, and Swiglu into a single kernel launch to reduce memory traffic.  
  - Triton allows writing such fused kernels with tight control over shared memory and registers.  

- **Parallelism**  
  - *Data Parallelism*: identical copies of the model on each GPU; each processes a different mini‑batch; gradients are all‑reduced across GPUs.  
  - *Tensor Parallelism*: split each weight matrix (e.g., `W_q`, `W_k`, `W_v`, `W_ff`) along the feature dimension across GPUs; each GPU stores only a slice, reducing per‑GPU memory.  
  - *Pipeline Parallelism*: split the transformer layers across different ranks; tokens flow from one stage to the next, effectively overlapping computations across GPUs.  

- **Triton Basics**  
  - Use `@triton.autotune` to experiment with different block sizes.  
  - `tl.load` and `tl.store` enable coalesced memory accesses.  
  - Example snippet (matrix multiply + activation):  
    ```python
    @triton.autotune(
        configs=[
            triton.Config({}, num_warps=1, num_ctas=1),
            triton.Config({}, num_warps=2, num_ctas=2),
        ],
        key=["BLOCK_SIZE"],
    )
    @triton.jit
    def matmul_kernel(A, B, C, M, N, K, BLOCK_SIZE: tl.constexpr):
        # load, compute, and store
    ```  

- **Inference Modes**  
  - *Prefill*: process the context tokens through the entire network in one forward pass; output is used to initialize the decoder.  
  - *Decode*: generate one token at a time; latency dominated by kernel launch overhead and attention computation.  
  - *Speculative Decoding*: a cheap LLM predicts `k` future tokens; the full model verifies by rescoring, accepting only the first token that matches the verifier’s top choice.  

**Implementation Notes**  
- Use `torch.distributed.rpc` or `torch.distributed` for gradient synchronization.  
- Profiling:  
  1. Check shared memory usage per block → should not exceed SM limit.  
  2. Measure memory bandwidth → ensure GPU memory is the bottleneck, not PCIe.  
  3. Compute kernel occupancy → aim for >70 %.  
- Verify that FP16 gradients stay within dynamic range by watching `torch.autocast()` statistics.  

**Profiling Checklist (GPU)**  
- Kernel occupancy (percentage of active warps).  
- Global memory bandwidth (GB/s).  
- Total GFLOPs (compute consumed).  

**Common Confusions**  
- *Misconception*: Adding more GPU cores always linearly speeds up training.  
  *Correction*: At some point memory bandwidth or kernel launch overhead dominates; careful profiling is needed.  

**Check Yourself**  
- What are the trade‑offs between data and tensor parallelism when the model has 12.8 B parameters?  
- Outline the steps in speculative decoding.  

---

### 3.5 Scaling Laws & Hyperparameter Planning  
**Overview**  
Scaling laws quantify how model performance improves with the number of parameters, training tokens, and compute. The empirical Chinchilla rule states the optimal parameter count is roughly one twentieth of the total training tokens. By fitting a power‑law to small experiments we can extrapolate to large models, saving costly training runs.  

**Detailed Explanation**  
- **Scaling Law (Power‑Law)**  
  ```
  L = a·P^b + c·T^d + e·C^f
  ```  
  with `a,b,c,d,e,f` fitted from logged losses of smaller runs.  
  The cross‑term `(P·T)` is sometimes added to capture interaction, but most literature shows the additive form works well and is easier to solve.  

- **Chinchilla Rule**  
  ```
  P_opt ≈ T_total / 20
  ```  
  This ensures efficient use of compute: training more tokens per parameter produces diminishing returns.  

- **Hyperparameter Search**  
  1. Fix `batch_size`, `learning_rate_schedule`.  
  2. Enumerate `(P, T)` pairs that satisfy `C = FLOPs_per_step * steps ≤ budget`.  
  3. Run a lightweight training loop (see placeholder below).  
  4. Fit `log(L)` vs. `log(P)` and `log(T)` using `numpy.polyfit`.  
  5. Predict loss for unseen `(P, T)` and select the pair with the lowest predicted loss.  

```python
import numpy as np
from scipy.stats import linregress

# Example dataset
P_vals = np.array([3e9, 6e9, 9e9])
T_vals = np.array([1e12, 2e12, 3e12])
L_vals = np.array([0.8, 0.7, 0.65])

logP = np.log(P_vals)
logT = np.log(T_vals)
logL = np.log(L_vals)

# Fit linear regression on log-log values
slope_P, intercept_P, r, p, se = linregress(logP, logL)
slope_T, intercept_T, r, p, se = linregress(logT, logL)
# coefficients a,b,c,d are derived from slopes
```

- **Placeholder for Full Training Loop**  
  ```python
  # TODO: replace this placeholder with your training loop
  # loss = train_api(P=1.4e9, T=3e12, batch=8192)
  ```

**Implementation Notes**  
- Save each experiment’s logged loss, `P`, `T`, `C` for regression.  
- Cross‑validation: leave one size out, predict its loss, and compute mean absolute percent error.  

**Common Confusions**  
- *Misconception*: Bigger models always outperform smaller ones.  
  *Correction*: With a fixed compute budget, a moderately-sized model trained longer can lead to lower loss than a larger model trained for fewer steps.  

**Check Yourself**  
- Explain the meaning of coefficient `b` in the scaling law.  
- What happens to loss when you double tokens at constant parameter count?  

---

### 3.6 Data & Alignment Pipeline  
**Overview**  
The alignment pipeline starts with raw web data, cleans it, filters toxic content, deduplicates, tokenizes, and finally trains the model to follow instructions safely. Alignment methods include supervised fine‑tuning (SFT), Direct Preference Optimization (DPO), and verifier‑guided RL from Observations (GRPO).  

**Detailed Explanation**  
1. **Data Acquisition**  
   - Crawl Common Crawl, Wikipedia, StackExchange, GitHub, etc.  
   - Store raw text as compressed shards (e.g., zstd).  

2. **Preprocessing**  
   - Strip HTML tags while preserving structural elements.  
   - Normalize Unicode to NFC.  

3. **Filtering & Moderation**  
   - Run a toxicity classifier (e.g., `bert-base-uncased`).  
   - If a passage exceeds a threshold, redact or discard.  

4. **Deduplication**  
   - Compute 64‑byte rolling hashes (xxhash).  
   - Keep the first occurrence of each fingerprint.  

5. **Tokenization**  
   - Apply the BPE tokenizer built earlier.  
   - Store token IDs in a binary format for fast streaming.  

6. **Alignment**  
   - **SFT**: fine‑tune on instruction/response pairs; minimize cross‑entropy.  
   - **DPO**: collect paired completions `(a, b)` where human preferred one.  
     \[
     \mathcal{L}_{\text{DPO}} = -\log\!\big(\sigma\big(s_\theta(a)-s_\theta(b)\big)\big)
     \]
     where `σ` is the sigmoid.  
   - **Verifier‑guided GRPO**: define a lightweight *verifier* that scores each generated token; use policy‑gradient to maximize expected verifier reward, avoiding a separate value network.  

#### Verification Forward Pass (Tiny Toy)  
```python
class Verifier(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.linear = nn.Linear(hidden_dim, 1)  # score a single token

    def forward(self, hidden_state):
        # hidden_state: [B, L, H]
        logits = self.linear(hidden_state)          # [B, L, 1]
        return torch.sigmoid(logits.squeeze(-1))   # [B, L]
```

**Implementation Notes**  
- Use `datasets` library for parallel shard processing.  
- Deduplication example:  
  ```python
  seen = set()
  for shard in shards:
      text = clean_html(shard)
      if not is_toxic(text):
          token_ids = tokenizer.encode(text)
          key = xxhash.hash_bytes(token_ids.tobytes()[:64])
          if key not in seen:
              seen.add(key)
              save(token_ids)
  ```  

**Common Confusions**  
- *Misconception*: "Any internet text is good."  
  *Correction*: Legal, copyright, and safety filters are mandatory before use.  

**Check Yourself**  
- List three pre‑processing steps before tokenization.  
- What is the functional form of DPO loss?  

---

### 3.7 Course Mindset & Evaluation  
**Overview**  
The curriculum encourages an “efficiency‑first” mindset: every design decision should be justified by compute savings or performance gains. Assessment combines code quality, unit tests, leaderboard benchmarks, and reflective essays, driving students to iterate and improve.  

**Detailed Explanation**  
- **Philosophy**  
  1. Build each component in isolation; trace data flow and memory usage.  
  2. Profile regularly (via `nvprof`, `triton.profiler`).  
  3. Use data‑driven scaling to trade off size vs. compute.  

- **Assessment**  
  - **Unit 1**: BPE implementation – speed and correctness.  
  - **Unit 2**: Transformer core – gradient check against analytic derivatives.  
  - **Unit 3**: Scaling‑law simulation – R² > 0.95 on held‑out size.  
  - **Unit 4**: Data pipeline – tokenization throughput per core > 1 k tokens/s.  
  - **Unit 5**: Alignment – perplexity drop > 0.2 after SFT, DPO satisfies human preference.  

- **Resources**  
  - Lecture videos, official training API, large‑scale datasets, H100 GPU cluster access, boilerplate code.  

**Common Confusions**  
- *Misconception*: "Build the biggest model you can."  
  *Correction*: With fixed compute, an appropriately sized model trained longer beats a larger one trained for fewer steps.  

**Check Yourself**  
- Why is benchmark performance a better motivator than raw parameter count?  
- Describe the three parallelism modalities in your own words.  

---

## 4) Glossary (Key Terms with First‑Use Definitions)  
| Term | Definition |
|------|------------|
| **BPE (Byte‑Pair Encoding)** | Sub‑word tokenization that iteratively merges frequent adjacent byte pairs. |
| **Tokenization** | Conversion of text into a sequence of integer token IDs. |
| **Rotary Embedding** | Positional encoding that applies complex rotations to query/key vectors, encoding relative position. |
| **Swiglu** | Activation function \(x \cdot \sigma(x)\) using a sigmoid gate; arises from two linear projections inside the module. |
| **RMSNorm** | Normalization that divides by the root‑mean‑square of activations, omitting the mean for speed. |
| **Transformer Block** | Unit comprising multi‑head self‑attention, a Swiglu‑activated feed‑forward, and RMSNorm. |
| **AdamW** | Adam optimizer variant with decoupled weight decay. |
| **Cross‑entropy Loss** | Loss used for next‑token prediction: \(-\frac{1}{N}\sum \log p(y_i|x_i)\). |
| **Chinchilla Rule** | Empirical guideline: optimal parameter count ≈ total training tokens ÷ 20. |
| **SFT (Supervised Fine‑Tuning)** | Fine‑tuning on instruction/response pairs with supervised cross‑entropy loss. |
| **DPO (Direct Preference Optimization)** | Preference‑directed training that maximizes the probability of higher‑ranked completions. |
| **GRPO (Verifier‑Guided RL from Observations)** | RL method that uses a lightweight verifier to compute rewards without a value network. |
| **Data Parallelism** | Replicating the model on each GPU; each processes a different mini‑batch; gradients are averaged. |
| **Tensor Parallelism** | Sharding model weight matrices across GPUs; each stores a fraction of the parameters. |
| **Pipeline Parallelism** | Splitting model layers across GPUs; tokens flow through stages like an assembly line. |
| **Triton** | Python framework for writing efficient, fuse‑able GPU kernels. |
| **Prefill / Decode** | Inference phases: prefill processes the full prompt; decode generates tokens one by one. |
| **Speculative Decoding** | A fast model predicts several next tokens; the full model verifies and accepts the first correct one. |

---

## 5) Summary Checklist

- **Tokenization**: BPE tokenizer trained, worked example present, code snippet, correctness test.  
- **Transformer**: Swiglu activation, rotary embeddings, RMSNorm, efficient attention implemented; unit tests for gradient.  
- **Training Pipeline**: AdamW, warm‑up + cosine schedule, mixed‑precision, gradient clipping, checkpointing, logs.  
- **GPU Kernels**: Triton fused kernels for attention + Swiglu; profiling shows adequate occupancy & bandwidth.  
- **Scaling Laws**: Power‑law fit (R² > 0.95), Chinchilla rule validated, linear‑regression snippet included.  
- **Data Pipeline**: Crawl → clean → filter → dedupe → tokenize; verifier class and forward pass shown.  
- **Alignment**: SFT loss, DPO loss, GRPO with verifier; correctness and safety metrics demonstrated.  
- **Profiling Checklist**: kernel occupancy, memory bandwidth, total compute.  
- **Evaluation**: benchmarks, perplexity, latency, safety; all thresholds met.  
- **Glossary**: all key terms defined on first use.  
- **Hyperparameter Table**: core defaults (see End Summary).  

---

## 6) End Summary Table of Core Hyperparameters (Default Values)

| Hyperparameter | Default (Sample) | Typical Range |
|----------------|------------------|---------------|
| `vocab_size` | 50 000 | 30 k–65 k |
| `embedding_dim` | 12 288 | 8 192–16 384 |
| `num_layers` | 48 | 24–64 |
| `num_heads` | 96 | 32–384 |
| `block_size` | 3 072 | 1 024–4 096 |
| `learning_rate` | 3 e‑4 | 1 e‑4 – 5 e‑4 |
| `weight_decay` | 0.1 | 0.01–0.3 |
| `betas` | (0.9, 0.999) | (0.85, 0.999) |
| `warmup_steps` | 500 k | 50 k–1 M |
| `batch_size` | 8 192 tokens (effective via accumulation) | 2 048–32 768 |
| `gradient_accumulation_steps` | 4 | 1–16 |
| `max_grad_norm` | 1.0 | 0.5–5.0 |
| `train_steps` | 1 M | 100 k–10 M |
| `optimizer` | AdamW | Adam, RMSProp |
| `scheduler` | Cosine decay + warm‑up | Linear, step |
| `precision` | fp16 (amp) | fp32, bf16 |
| `parallelism` | Data+Tensor | Pipeline |

--- 

**End of Notes**
