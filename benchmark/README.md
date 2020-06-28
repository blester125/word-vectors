# Benchmark Results


We can see from our bench marking that the Leader format if the fastest and that our library is faster then Gensim.

```
Statistically Significant: GloVe 6B w2v vs Leader
	t: -8.256250993845377
	p: 3.47772855575113e-05
	We reject the null hypothesis, therefore:
		 "Leader is faster than w2v"

Statistically Significant: GloVe 27B w2v vs Leader
	t: -32.92055416602123
	p: 7.906451860619565e-10
	We reject the null hypothesis, therefore:
		 "Leader is faster than w2v"

Statistically Significant: GloVe 42B w2v vs Leader
	t: -7.963909960090788
	p: 4.511226572800015e-05
	We reject the null hypothesis, therefore:
		 "Leader is faster than w2v"

Statistically Significant: GloVe 840B w2v vs Leader
	t: -9.033798819477678
	p: 1.8027337125810306e-05
	We reject the null hypothesis, therefore:
		 "Leader is faster than w2v"

Statistically Significant: FastText Wiki w2v vs Leader
	t: -7.548356617620479
	p: 6.618453892829221e-05
	We reject the null hypothesis, therefore:
		 "Leader is faster than w2v"

Statistically Significant: FastText Crawl w2v vs Leader
	t: -12.731618814866351
	p: 1.3639525328896334e-06
	We reject the null hypothesis, therefore:
		 "Leader is faster than w2v"

Statistically Significant: Google News w2v vs Leader
	t: -63.52979208811108
	p: 4.1908543163704255e-12
	We reject the null hypothesis, therefore:
		 "Leader is faster than w2v"

Statistically Significant: GloVe 6B, word2vec text, word-vectors vs gensim
	t: -57.109414279652874
	p: 9.811288067304373e-12
	We reject the null hypothesis, therefore:
		 "word-vectors is faster than gensim"

Statistically Significant: GloVe 6B, word2vec binary, word-vectors vs gensim
	t: -8.21730116653506
	p: 3.5988397917038216e-05
	We reject the null hypothesis, therefore:
		 "word-vectors is faster than gensim"

Statistically Significant: GloVe 27B, word2vec text, word-vectors vs gensim
	t: -48.93331732007736
	p: 3.3664001979213617e-11
	We reject the null hypothesis, therefore:
		 "word-vectors is faster than gensim"

Statistically Significant: GloVe 27B, word2vec binary, word-vectors vs gensim
	t: -3.8099022410519074
	p: 0.005163351919086772
	We reject the null hypothesis, therefore:
		 "word-vectors is faster than gensim"

Statistically Significant: GloVe 42B, word2vec text, word-vectors vs gensim
	t: -266.90526279798985
	p: 4.346957154548019e-17
	We reject the null hypothesis, therefore:
		 "word-vectors is faster than gensim"

Statistically Significant: GloVe 42B, word2vec binary, word-vectors vs gensim
	t: -11.672966833947159
	p: 2.6456049550214714e-06
	We reject the null hypothesis, therefore:
		 "word-vectors is faster than gensim"

Statistically Significant: GloVe 840B, word2vec text, word-vectors vs gensim
	t: -159.28266102447128
	p: 2.7000828284169086e-15
	We reject the null hypothesis, therefore:
		 "word-vectors is faster than gensim"

Statistically Significant: GloVe 840B, word2vec binary, word-vectors vs gensim
	t: -11.965985393106985
	p: 2.1907416561334928e-06
	We reject the null hypothesis, therefore:
		 "word-vectors is faster than gensim"

Statistically Significant: FastText Wiki, word2vec text, word-vectors vs gensim
	t: -96.09463692984482
	p: 1.5355745262208231e-13
	We reject the null hypothesis, therefore:
		 "word-vectors is faster than gensim"

Statistically Significant: FastText Wiki, word2vec binary, word-vectors vs gensim
	t: -7.923076923076918
	p: 4.681037260356795e-05
	We reject the null hypothesis, therefore:
		 "word-vectors is faster than gensim"

Statistically Significant: FastText Crawl, word2vec text, word-vectors vs gensim
	t: -133.9081607906148
	p: 1.0815940389259812e-14
	We reject the null hypothesis, therefore:
		 "word-vectors is faster than gensim"

Statistically Significant: FastText Crawl, word2vec binary, word-vectors vs gensim
	t: -11.979292478099957
	p: 2.172268196934171e-06
	We reject the null hypothesis, therefore:
		 "word-vectors is faster than gensim"

Statistically Significant: Google News, word2vec text, word-vectors vs gensim
	t: -15.27030907765855
	p: 3.3549114879400533e-07
	We reject the null hypothesis, therefore:
		 "word-vectors is faster than gensim"

Statistically Significant: Google News, word2vec binary, word-vectors vs gensim
	t: -5.618507943013474
	p: 0.0004993619677095291
	We reject the null hypothesis, therefore:
		 "word-vectors is faster than gensim"
```
