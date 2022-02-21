var store = [{
        "title": "Paper_uai2016",
        "excerpt":"Faster stochastic variational inference using proximal-gradient methods with general divergence functions accepted at UAI2016! ","categories": [],
        "tags": [],
        "url": "/news/paper_uai2016/",
        "teaser":null},{
        "title": "Paper_aistats2017",
        "excerpt":"Conjugate-computation variational inference accepted at AIStats2017! ","categories": [],
        "tags": [],
        "url": "/news/paper_aistats2017/",
        "teaser":null},{
        "title": "Paper_iclr2018",
        "excerpt":"Variational message passing with structured inference networks accepted at ICLR2018! ","categories": [],
        "tags": [],
        "url": "/news/paper_iclr2018/",
        "teaser":null},{
        "title": "Paper_icml2018",
        "excerpt":"Fast and scalable bayesian deep learning by weight-perturbation in Adam accepted at ICML2018! ","categories": [],
        "tags": [],
        "url": "/news/paper_icml2018/",
        "teaser":null},{
        "title": "Workshop_icml2019",
        "excerpt":"New workshop paper on Stein’s Lemma for the Reparameterization Trick with Exponential Family Mixtures out. Will be presented at the Stein’s Method in Machine Learning and Statistics Workshop at ICML2019. ","categories": [],
        "tags": [],
        "url": "/news/workshop_icml2019/",
        "teaser":null},{
        "title": "Paper_icml2019",
        "excerpt":"Fast and simple natural-gradient variational inference with mixture of exponential-family approximations accepted at ICML2019! ","categories": [],
        "tags": [],
        "url": "/news/paper_icml2019/",
        "teaser":null},{
        "title": "Paper_icml2020",
        "excerpt":"Handling the positive-definite constraint in the bayesian learning rule accepted at ICML2020, see the ICML talk. ","categories": [],
        "tags": [],
        "url": "/news/paper_icml2020/",
        "teaser":null},{
        "title": "Paper_icml2021",
        "excerpt":"Tractable structured natural gradient descent using local parameterizations accepted at ICML2021! ","categories": [],
        "tags": [],
        "url": "/news/paper_icml2021/",
        "teaser":null},{
        "title": "Workshop_icml2021",
        "excerpt":"New workshop paper on Structured second-order methods via natural gradient descent out. Will be presented at the Beyond first-order methods in ML systems Workshop at ICML2021, see the spotlight talk. ","categories": [],
        "tags": [],
        "url": "/news/workshop_icml2021/",
        "teaser":null},{
        "title": "News",
        "excerpt":"I held a reading group on geometric structures in machine learning at the UBC Machine Learning Reading Group. ","categories": [],
        "tags": [],
        "url": "/news/news/",
        "teaser":null},{
        "title": "GD and NGD",
        "excerpt":"%matplotlib inlinefrom jax.config import config; config.update(\"jax_enable_x64\", True)import jax.numpy as jnpfrom jax import grad, jit, value_and_gradimport numpy as npfrom matplotlib import pyplot as pltfrom matplotlib import ticker, colors@jitdef loss_lik(mu,v): b1 = 0.5; b2 = 0.01; a1 = 2.0; a2 = 5.0; ls = b1*(mu**2+v-2.0*a1*mu+a1**2)+b2*((mu**3+3.0*mu*v)-3.0*a2*(mu**2+v)+3.0*(a2**2)*mu-a2**3)+4.0/v return ls@jitdef loss_pre(params): (mu,s) = params return...","categories": [],
        "tags": ["Natural Gradient Descent","Python"],
        "url": "/posts/2022/01/notebooks/",
        "teaser":null},{
        "title": "Structured Natural Gradient Descent (ICML 2021)",
        "excerpt":"More about this work [1]: (Youtube) talk, extended paper, short paper,poster Introduction Motivation Many problems in optimization, search, and inference can be solved via natural-gradient descent (NGD) Structures play an essential role in Preconditioners of first-order and second-order optimization, gradient-free search. Covariance matrices of variational Gaussian inference [2]Natural-gradient descent on...","categories": [],
        "tags": ["Natural Gradient Descent","Information Geometry","Matrix Lie Groups","Exponential Family"],
        "url": "/posts/2021/07/ICML/",
        "teaser":null},{
        "title": "Part I: Smooth Manifolds with the Fisher-Rao Metric",
        "excerpt":"Goal This blog post focuses on the Fisher-Rao metric, which gives rise to the Fisher information matrix (FIM). We will introduce the following concepts, useful to ensure non-singular FIMs: Regularity conditions and intrinsic parameterizations of a distribution Dimensionality of a smooth manifoldThe discussion here is informal and focuses on more...","categories": [],
        "tags": ["Natural Gradient Descent","Information Geometry","Riemannian Manifold"],
        "url": "/posts/2021/09/Geomopt01/",
        "teaser":null},{
        "title": "Part II: Derivation of Natural-gradients",
        "excerpt":"Warning: working in Progress (incomplete) Goal This blog post focuses on natural-gradients, which are known as Riemannian gradients with the Fisher-Rao metric. We will discuss the following concepts to derive natural-gradients: Direction derivatives in a manifold Weighted norm induced by the Fisher-Rao metric Rimannian steepest direction Space of natural-gradients evaluated...","categories": [],
        "tags": ["Natural Gradient Descent","Information Geometry","Riemannian Manifold"],
        "url": "/posts/2021/10/Geomopt02/",
        "teaser":null},{
        "title": "Part III: Invariance of Natural-Gradients",
        "excerpt":"Warning: working in Progress (incomplete) Goal This blog post should help readers to understand the invariance of natural-gradients.We will also discuss why the Euclidean steepest direction is NOT invariant. We will give an informal introduction with a focus on high level of ideas. Parameter Transformation and Invariance In Part II,...","categories": [],
        "tags": ["Natural Gradient Descent","Information Geometry","Riemannian Manifold"],
        "url": "/posts/2021/11/Geomopt03/",
        "teaser":null},{
        "title": "Part IV: Natural and Riemannian  Gradient Descent",
        "excerpt":"Warning: working in Progress (incomplete) Goal This blog post should help readers to understand natural-gradient descent and Riemannian gradient descent.We also discuss some invariance property of natural-gradient descent, Riemannian gradient descent, and Newton’s method. We will give an informal introduction with a focus on high level of ideas. Two kinds...","categories": [],
        "tags": ["Natural Gradient Descent","Information Geometry","Riemannian Manifold"],
        "url": "/posts/2021/11/Geomopt04/",
        "teaser":null},{
        "title": "Part V: Efficient Natural-gradient Methods for Exponential Family",
        "excerpt":"Warning: working in Progress (incomplete) Goal This blog post should show that we can efficiently implement natural-gradient methods in many cases. We will give an informal introduction with a focus on high level of ideas. Exponential Family An exponential family takes the following (canonical) form as$$\\begin{aligned}p(\\mathbf{w}|\\mathbf{\\eta}) = h_\\eta(\\mathbf{w}) \\exp( \\langle...","categories": [],
        "tags": ["Natural Gradient Descent","Information Geometry","Riemannian Manifold","Exponential Family"],
        "url": "/posts/2021/12/Geomopt05/",
        "teaser":null},{
        "title": "Part VI: Handling Parameter Constraints of Exponential Family In Natural-gradient Methods",
        "excerpt":"Warning: working in Progress (incomplete) Goal In this blog post, we discuss about how to handle parameter constraints of exponential family. Handling Parameter Constraints Recall that in Part IV, we discuss the many faces of NGD in unconstrained cases. These methods could also be exteneded in constrained cases to handle...","categories": [],
        "tags": ["Natural Gradient Descent","Information Geometry","Riemannian Manifold","Exponential Family"],
        "url": "/posts/2021/12/Geomopt06/",
        "teaser":null},{
        "title": "Paper Title Number 1",
        "excerpt":"This paper is about the number 1. The number 2 is left for future work. Download paper here Recommended citation: Your Name, You. (2009). “Paper Title Number 1.” Journal 1. 1(1). ","categories": [],
        "tags": [],
        "url": "/publication/2009-10-01-paper-title-number-1",
        "teaser":null},{
        "title": "Paper Title Number 2",
        "excerpt":"This paper is about the number 2. The number 3 is left for future work. Download paper here Recommended citation: Your Name, You. (2010). “Paper Title Number 2.” Journal 1. 1(2). ","categories": [],
        "tags": [],
        "url": "/publication/2010-10-01-paper-title-number-2",
        "teaser":null},{
        "title": "Paper Title Number 3",
        "excerpt":"This paper is about the number 3. The number 4 is left for future work. Download paper here Recommended citation: Your Name, You. (2015). “Paper Title Number 3.” Journal 1. 1(3). ","categories": [],
        "tags": [],
        "url": "/publication/2015-10-01-paper-title-number-3",
        "teaser":null}]
