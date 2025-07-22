---
permalink: /
title: 
  zh: "关于我"
  en: "About Me"
excerpt: 
  zh: "个人简介"
  en: "Personal Profile"
author_profile: true
redirect_from: 
  - /about/
  - /about.html
---

<span class='anchor' id='about-me'></span>

<!-- 语言切换按钮 -->
<div style="margin: 1em; text-align: left;">
  <button id="lang-toggle" onclick="toggleLang()" style="
    padding: 0.6em 1.2em;
    font-size: 1em;
    font-weight: 500;
    color: #ffffff;
    background-color: #3b82f6;
    border: none;
    border-radius: 6px;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 2px 4px rgba(59, 130, 246, 0.2);
  ">
    <span id="btn-text">切换为中文</span>
  </button>
</div>

{% capture en_intro %}{%  include_relative includes/intro.md %}{% endcapture %}
{% capture en_news %}{%   include_relative includes/news.md %}{% endcapture %}
{% capture en_pub %}{%    include_relative includes/pub.md %}{% endcapture %}
{% capture en_honers %}{% include_relative includes/honers.md %}{% endcapture %}
{% capture en_others %}{% include_relative includes/others.md %}{% endcapture %}

<!-- 同时加载中英文内容，用类名区分 -->
<div class="lang-content lang-en">
{{ en_intro  | markdownify }}
{{ en_news   | markdownify }}
{{ en_pub    | markdownify }}
{{ en_honers | markdownify }}
{{ en_others | markdownify }}
</div>

{% capture zh_intro %}{%  include_relative includes_zh/intro.md %}{% endcapture %}
{% capture zh_news %}{%   include_relative includes_zh/news.md %}{% endcapture %}
{% capture zh_pub %}{%    include_relative includes_zh/pub.md %}{% endcapture %}
{% capture zh_honers %}{% include_relative includes_zh/honers.md %}{% endcapture %}
{% capture zh_others %}{% include_relative includes_zh/others.md %}{% endcapture %}

<div class="lang-content lang-zh" style="display: none;">
  {{ zh_intro  | markdownify }}
  {{ zh_news   | markdownify }}
  {{ zh_pub    | markdownify }}
  {{ zh_honers | markdownify }}
  {{ zh_others | markdownify }}
</div>

<script>
  // 页面加载时初始化显示
  document.addEventListener('DOMContentLoaded', () => {
    const urlParams = new URLSearchParams(window.location.search);
    const currentLang = urlParams.get('lang') || 'en';
    const btnText = document.getElementById('btn-text');
    
    // 显示当前语言内容
    document.querySelector('.lang-en').style.display = currentLang === 'en' ? 'block' : 'none';
    document.querySelector('.lang-zh').style.display = currentLang === 'zh' ? 'block' : 'none';
    btnText.textContent = currentLang === 'en' ? '切换为中文' : 'Switch to English';
  });

  // 切换语言逻辑
  function toggleLang() {
    const urlParams = new URLSearchParams(window.location.search);
    const currentLang = urlParams.get('lang') || 'en';
    const newLang = currentLang === 'en' ? 'zh' : 'en';
    
    // 更新URL参数（不刷新页面）
    urlParams.set('lang', newLang);
    const newUrl = window.location.pathname + (urlParams.toString() ? '?' + urlParams.toString() : '');
    window.history.pushState({}, '', newUrl);
    
    // 切换内容显示
    document.querySelector('.lang-en').style.display = newLang === 'en' ? 'block' : 'none';
    document.querySelector('.lang-zh').style.display = newLang === 'zh' ? 'block' : 'none';
    document.getElementById('btn-text').textContent = newLang === 'en' ? '切换为中文' : 'Switch to English';
  }
</script>

<a href=""><img src="https://s05.flagcounter.com/count2/XcvZ/bg_FFFEFA/txt_000000/border_CCCCCC/columns_2/maxflags_10/viewers_0/labels_0/pageviews_0/flags_0/percent_0/" alt="Flag Counter" border="0"></a>

<!-- <a href="https://clustrmaps.com/site/1c76b"  title="ClustrMaps"><img src="//www.clustrmaps.com/map_v2.png?d=xkfYpF7jdZxG0K1rsuHQGjnlMfNBPUUq7pLSiLA0vz0&cl=ffffff" /></a> -->

<div style="width: 100%; max-width: 200px; height: 0; position: relative; margin: auto;">
  <script type="text/javascript" id="clstr_globe" src="//clustrmaps.com/globe.js?d=xkfYpF7jdZxG0K1rsuHQGjnlMfNBPUUq7pLSiLA0vz0"></script>
</div>
