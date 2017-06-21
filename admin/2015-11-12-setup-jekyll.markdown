---
layout: post
title:  "Setting up Jekyll Static Blog"
date:   2015-11-12 06:13:33
categories: jekyll update
---

* For Debian, just install jekyll with apt.  
{% highlight shell%}
apt install jekyll
{% endhighlight %}

* Initialize the blog directory with jekyll.  
{% highlight shell%}
jekyll new myblog
{% endhighlight %}

* Edit the `_config.yml` and markdown files under `_posts`.  
{% highlight shell%}
vim _posts
{% endhighlight %}

* Build the site. Generated site will be placed under directory `_site`.  
{% highlight shell%}
jekyll build
{% endhighlight %}

* Check the result.    
{% highlight shell%}
jekyll server
firefox localhost:4000
{% endhighlight %}

* Push the site onto remote host.  
{% highlight ruby %}
rsync -acv _site/ me@remotehost:~/public_html/
{% endhighlight %}
