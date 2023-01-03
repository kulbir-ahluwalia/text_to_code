"use strict";(()=>{var le=Object.defineProperty;var t=(O,A)=>le(O,"name",{value:A,configurable:!0});(globalThis.webpackChunk=globalThis.webpackChunk||[]).push([["app_assets_modules_github_soft-navigate_ts"],{41307:(O,A,l)=>{l.d(A,{ZP:()=>N,y0:()=>ae});var h=l(6170),i=l(54293),u=l(61161),d=l(12981),E=l(45922),S=l(36192);const y=20;let o,w=null;function g(s,e,r){return s.dispatchEvent(new CustomEvent(e,{bubbles:!0,cancelable:!0,detail:r}))}t(g,"dispatch");async function N(s){const e={push:!0,replace:!1,type:"GET",dataType:"html",scrollTo:0,...s};e.requestUrl=e.url;const n=f(e.url).hash,c=e.container,C=b(c);o||(o={id:P(),url:window.location.href,title:document.title,container:C,fragment:e.fragment},(0,h.lO)(o,o.title,o.url)),w?.abort();const{signal:R}=w=new AbortController;e.push===!0&&e.replace!==!0&&(ne(o.id,a(c)),(0,h.qA)(null,"",e.requestUrl)),g(c,"pjax:start",{url:e.url}),g(c,"pjax:send");let p;const M=se();try{p=await fetch(e.url,{signal:R,method:e.type,body:e.data,headers:{Accept:"text/html","X-PJAX":"true","X-PJAX-Container":C,"X-Requested-With":"XMLHttpRequest","X-PJAX-VERSION":M.pjax??"","X-PJAX-CSP-VERSION":M.csp??"","X-PJAX-CSS-VERSION":M.css??"","X-PJAX-JS-VERSION":M.js??""}})}catch{p=void 0}if(!p||!p.ok){const X=g(c,"pjax:error");if(e.type==="GET"&&X){const G=p&&p.headers.get("X-PJAX-URL"),ie=G?f(G).href:e.requestUrl;(0,E.b)({pjaxFailureReason:"response_error",requestUrl:e.requestUrl}),U(ie)}g(c,"pjax:complete"),g(c,"pjax:end");return}const I=o,v=re(),j=p.headers.get("X-PJAX-Version"),T=await p.text(),x=Y(T,p,e),{contents:K}=x,Q=f(x.url);if(n&&(Q.hash=n,x.url=Q.href),v&&j&&v!==j){g(c,"pjax:hardLoad",{reason:"version_mismatch"}),(0,E.b)({pjaxFailureReason:"version_mismatch",requestUrl:e.requestUrl}),U(x.url);return}if(!K){g(c,"pjax:hardLoad",{reason:"missing_response_body"}),(0,E.b)({pjaxFailureReason:"missing_response_body",requestUrl:e.requestUrl}),U(x.url);return}o={id:e.id!=null?e.id:P(),url:x.url,title:x.title,container:C,fragment:e.fragment},(e.push===!0||e.replace===!0)&&(0,h.lO)(o,x.title,x.url);const $=document.activeElement,ce=e.container!=null&&e.container.contains($);if($ instanceof HTMLElement&&ce)try{$.blur()}catch{}x.title&&(document.title=x.title),g(c,"pjax:beforeReplace",{contents:K,state:o,previousState:I}),D(c,K),(0,i.b)(),(0,i.o)();const z=c.querySelector("input[autofocus], textarea[autofocus]");z&&document.activeElement!==z&&z.focus(),x.scripts&&ee(x.scripts),x.stylesheets&&te(x.stylesheets);let Z=e.scrollTo;if(n){const X=(0,u.Kt)(document,n);X&&(Z=X.getBoundingClientRect().top+window.pageYOffset)}typeof Z=="number"&&window.scrollTo(window.pageXOffset,Z),g(c,"pjax:success"),g(c,"pjax:complete"),g(c,"pjax:end")}t(N,"pjaxRequest");function U(s){o&&(0,h.lO)(null,"",o.url),window.location.replace(s)}t(U,"locationReplace");let q=!0;const B=window.location.href,_=window.history.state;_&&_.container&&(o=_),"state"in window.history&&(q=!1);function L(s){if((0,S.xc)())return;q||w?.abort();const e=o,r=s.state;let n=null;if(r&&r.container){if(q&&B===r.url)return;if(e){if(e.id===r.id)return;n=e.id<r.id?"forward":"back"}const[c,C,R]=k[r.id]||[],p=document.querySelector(c||r.container);if(p instanceof HTMLElement){e&&oe(n,e.id,a(p)),g(p,"pjax:popstate",{state:r,direction:n,cachedAt:R});const M={id:r.id,url:r.url,container:p,push:!1,fragment:r.fragment||"",scrollTo:!1};C?(g(p,"pjax:start"),o=r,r.title&&(document.title=r.title),g(p,"pjax:beforeReplace",{contents:C,state:r,previousState:e}),D(p,C),(0,i.b)(),(0,i.o)(),g(p,"pjax:end")):N(M),p.offsetHeight}else(0,E.b)({pjaxFailureReason:"no_container",requestUrl:e?.url}),U(location.href)}q=!1}t(L,"onPjaxPopstate");function P(){return new Date().getTime()}t(P,"uniqueId");function a(s){const e=s.cloneNode(!0);return[b(s),Array.from(e.childNodes),Date.now()]}t(a,"cloneContents");function f(s){const e=document.createElement("a");return e.href=s,e}t(f,"parseURL");function b(s){if(s.id)return`#${s.id}`;throw new Error("pjax container has no id")}t(b,"getContainerSelector");function m(s,e,r){let n=[];for(const c of s)c instanceof Element&&(c instanceof r&&c.matches(e)&&n.push(c),n=n.concat(Array.from(c.querySelectorAll(e))));return n}t(m,"findAll");function D(s,e){s.innerHTML="";for(const r of e)r!=null&&s.appendChild(r)}t(D,"replaceWithNodes");function W(s,e){const r=s.headers.get("X-PJAX-URL");return r?f(r).href:e}t(W,"resolveUrl");function Y(s,e,r){const n={url:W(e,r.requestUrl),title:""},c=/<html/i.test(s);if((e.headers.get("Content-Type")||"").split(";",1)[0].trim()!=="text/html")return n;let R,p;if(c){const v=s.match(/<head[^>]*>([\s\S.]*)<\/head>/i),j=s.match(/<body[^>]*>([\s\S.]*)<\/body>/i);R=v?Array.from((0,d.r)(document,v[0]).childNodes):[],p=j?Array.from((0,d.r)(document,j[0]).childNodes):[]}else R=p=Array.from((0,d.r)(document,s).childNodes);if(p.length===0)return n;const M=m(R,"title",HTMLTitleElement);n.title=M.length>0&&M[M.length-1].textContent||"";let I;if(r.fragment){if(r.fragment==="body")I=p;else{const v=m(p,r.fragment,Element);I=v.length>0?[v[0]]:[]}if(I.length&&(r.fragment==="body"?n.contents=I:n.contents=I.flatMap(v=>Array.from(v.childNodes)),!n.title)){const v=I[0];v instanceof Element&&(n.title=v.getAttribute("title")||v.getAttribute("data-title")||"")}}else c||(n.contents=p);if(n.contents){n.contents=n.contents.filter(function(T){return T instanceof Element?!T.matches("title"):!0});for(const T of n.contents)if(T instanceof Element)for(const x of T.querySelectorAll("title"))x.remove();const v=m(n.contents,"script[src]",HTMLScriptElement);for(const T of v)T.remove();n.scripts=v,n.contents=n.contents.filter(T=>v.indexOf(T)===-1);const j=m(n.contents,"link[rel=stylesheet]",HTMLLinkElement);for(const T of j)T.remove();n.stylesheets=j,n.contents=n.contents.filter(T=>!j.includes(T))}return n.title&&(n.title=n.title.trim()),n}t(Y,"extractContainer");function ee(s){const e=document.querySelectorAll("script[src]");for(const r of s){const{src:n}=r;if(Array.from(e).some(p=>p.src===n))continue;const c=document.createElement("script"),C=r.getAttribute("type");C&&(c.type=C);const R=r.getAttribute("integrity");R&&(c.integrity=R,c.crossOrigin="anonymous"),c.src=n,document.head&&document.head.appendChild(c)}}t(ee,"executeScriptTags");function te(s){const e=document.querySelectorAll("link[rel=stylesheet]");for(const r of s)Array.from(e).some(n=>n.href===r.href)||document.head&&document.head.appendChild(r)}t(te,"injectStyleTags");const k={},H=[],F=[];function ne(s,e){k[s]=e,F.push(s),V(H,0),V(F,y)}t(ne,"cachePush");function oe(s,e,r){let n,c;k[e]=r,s==="forward"?(n=F,c=H):(n=H,c=F),n.push(e);const C=c.pop();C&&delete k[C],V(n,y)}t(oe,"cachePop");function V(s,e){for(;s.length>e;){const r=s.shift();if(r==null)return;delete k[r]}}t(V,"trimCacheStack");function re(){for(const s of document.getElementsByTagName("meta")){const e=s.getAttribute("http-equiv");if(e&&e.toUpperCase()==="X-PJAX-VERSION")return s.content}return null}t(re,"findVersion");function J(s){return document.querySelector(`meta[http-equiv="${s}"]`)?.content??null}t(J,"pjaxMeta");function se(){return{pjax:J("X-PJAX-VERSION"),csp:J("X-PJAX-CSP-VERSION"),css:J("X-PJAX-CSS-VERSION"),js:J("X-PJAX-JS-VERSION")}}t(se,"findAllVersions");function ae(){return o}t(ae,"getState"),window.addEventListener("popstate",L)},54293:(O,A,l)=>{l.d(A,{b:()=>d,o:()=>E});var h=l(7739);const i={},u={};(async()=>(await h.x,i[document.location.pathname]=Array.from(document.querySelectorAll("head [data-pjax-transient]")),u[document.location.pathname]=Array.from(document.querySelectorAll("[data-pjax-replace]"))))(),document.addEventListener("pjax:beforeReplace",function(S){const y=S.detail.contents||[],o=S.target;for(let w=0;w<y.length;w++){const g=y[w];g instanceof Element&&(g.id==="pjax-head"?(i[document.location.pathname]=Array.from(g.children),y[w]=null):g.hasAttribute("data-pjax-replace")&&(u[document.location.pathname]||(u[document.location.pathname]=[]),u[document.location.pathname].push(g),o.querySelector(`#${g.id}`)||(y[w]=null)))}});function d(){const S=u[document.location.pathname];if(!!S)for(const y of S){const o=document.querySelector(`#${y.id}`);o&&o.replaceWith(y)}}t(d,"replaceCachedElements");function E(){const S=i[document.location.pathname];if(!S)return;const y=document.head;for(const o of document.querySelectorAll("head [data-pjax-transient]"))o.remove();for(const o of S)o.matches("title, script, link[rel=stylesheet]")?o.matches("link[rel=stylesheet]")&&y.append(o):(o.setAttribute("data-pjax-transient",""),y.append(o))}t(E,"replaceTransientTags")},36192:(O,A,l)=>{l.d(A,{AU:()=>w,DT:()=>q,F2:()=>y,HN:()=>E,Si:()=>o,aN:()=>S,ag:()=>P,q3:()=>g,rc:()=>L,wz:()=>U,xc:()=>d,xk:()=>B});var h=l(74395),i=l(49815);const u=h.session.adapter,d=t(()=>!(0,i.c)("PJAX_ENABLED"),"isTurboEnabled"),E=t(a=>a?.tagName==="TURBO-FRAME","isTurboFrame"),S=t(()=>{u.progressBar.setValue(0),u.progressBar.show()},"beginProgressBar"),y=t(()=>{u.progressBar.setValue(1),u.progressBar.hide()},"completeProgressBar"),o=t((a,f)=>{const b=new URL(a,window.location.origin),m=new URL(f,window.location.origin);return Boolean(m.hash)&&b.hash!==m.hash&&b.host===m.host&&b.pathname===m.pathname&&b.search===m.search},"isHashNavigation");function w(a,f){const b=a.split("/",3).join("/"),m=f.split("/",3).join("/");return b===m}t(w,"isSameRepo");async function g(){const a=document.head.querySelectorAll("link[rel=stylesheet]"),f=new Set([...document.styleSheets].map(m=>m.href)),b=[];for(const m of a)m.href===""||f.has(m.href)||b.push(N(m));await Promise.all(b)}t(g,"waitForStylesheets");const N=t((a,f=2e3)=>new Promise(b=>{const m=t(()=>{a.removeEventListener("error",m),a.removeEventListener("load",m),b()},"onComplete");a.addEventListener("load",m,{once:!0}),a.addEventListener("error",m,{once:!0}),setTimeout(m,f)}),"waitForLoad"),U=t(a=>{const f=a.querySelectorAll("[data-turbo-replace]"),b=[...document.querySelectorAll("[data-turbo-replace]")];for(const m of f){const D=b.find(W=>W.id===m.id);D&&D.replaceWith(m)}},"replaceElements"),q=t(a=>{for(const f of a.querySelectorAll("link[rel=stylesheet]"))document.head.querySelector(`link[href="${f.getAttribute("href")}"],
           link[data-href="${f.getAttribute("data-href")}"]`)||document.head.append(f)},"addNewStylesheets"),B=t(a=>{for(const f of a.querySelectorAll("script"))document.head.querySelector(`script[src="${f.getAttribute("src")}"]`)||_(f)},"addNewScripts"),_=t(a=>{const{src:f}=a,b=document.createElement("script"),m=a.getAttribute("type");m&&(b.type=m);const D=a.getAttribute("integrity");D&&(b.integrity=D,b.crossOrigin="anonymous"),b.src=f,document.head&&document.head.appendChild(b)},"executeScriptTag"),L=t(a=>{for(const f of a.querySelectorAll('meta[data-turbo-track="reload"]'))if(document.querySelector(`meta[http-equiv="${f.getAttribute("http-equiv")}"]`)?.content!==f.content)return!1;return!0},"matchingTrackedElements"),P=t(a=>{const f=a.querySelector("[data-turbo-head]")||a.head;return{title:f.querySelector("title")?.textContent,transients:[...f.querySelectorAll("[data-pjax-transient]")],bodyClasses:a.querySelector("meta[name=turbo-body-classes]")?.content}},"getTurboCacheNodes")},7739:(O,A,l)=>{l.d(A,{C:()=>i,x:()=>h});const h=function(){return document.readyState==="interactive"||document.readyState==="complete"?Promise.resolve():new Promise(u=>{document.addEventListener("DOMContentLoaded",()=>{u()})})}(),i=function(){return document.readyState==="complete"?Promise.resolve():new Promise(u=>{window.addEventListener("load",u)})}()},49815:(O,A,l)=>{l.d(A,{$:()=>S,c:()=>d});var h=l(15205);const i=(0,h.Z)(u);function u(){return(document.head?.querySelector('meta[name="enabled-features"]')?.content||"").split(",")}t(u,"enabledFeatures");const d=(0,h.Z)(E);function E(y){return i().indexOf(y)!==-1}t(E,"isEnabled");const S={isFeatureEnabled:d}},61161:(O,A,l)=>{l.d(A,{$z:()=>u,Kt:()=>h,Q:()=>i});function h(d,E=location.hash){return i(d,u(E))}t(h,"findFragmentTarget");function i(d,E){return E===""?null:d.getElementById(E)||d.getElementsByName(E)[0]}t(i,"findElementByFragmentName");function u(d){try{return decodeURIComponent(d.slice(1))}catch{return""}}t(u,"decodeFragmentValue")},6170:(O,A,l)=>{l.d(A,{Mw:()=>q,_C:()=>U,lO:()=>N,qA:()=>g,y0:()=>d});const h=[];let i=0,u;function d(){return u}t(d,"getState");function E(){try{return Math.min(Math.max(0,history.length)||0,9007199254740991)}catch{return 0}}t(E,"safeGetHistory");function S(){const _={_id:new Date().getTime(),...history.state};return o(_),_}t(S,"initializeState");function y(){return E()-1+i}t(y,"position");function o(_){u=_;const L=location.href;h[y()]={url:L,state:u},h.length=E(),window.dispatchEvent(new CustomEvent("statechange",{bubbles:!1,cancelable:!1}))}t(o,"setState");function w(){return new Date().getTime()}t(w,"uniqueId");function g(_,L,P){i=0;const a={_id:w(),..._};history.pushState(a,L,P),o(a)}t(g,"pushState");function N(_,L,P){const a={...d(),..._};history.replaceState(a,L,P),o(a)}t(N,"replaceState");function U(){const _=h[y()-1];if(_)return _.url}t(U,"getBackURL");function q(){const _=h[y()+1];if(_)return _.url}t(q,"getForwardURL"),u=S(),window.addEventListener("popstate",t(function(L){const P=L.state;if(!(!P||!P._id&&!P.turbo?.restorationIdentifier)){if(P.turbo?.restorationIdentifier){const a=P.turbo.restorationIdentifier;h[y()-1]?.state?.turbo?.restorationIdentifier===a?i--:i++}else P._id<(d()._id||NaN)?i--:i++;o(P)}},"onPopstate"),!0);let B;window.addEventListener("turbo:visit",_=>{_ instanceof CustomEvent&&(B=_.detail.action)}),window.addEventListener("turbo:load",()=>{B!=="restore"&&(i=0,N(history.state,"",""))}),window.addEventListener("hashchange",t(function(){if(E()>h.length){const L={_id:w()};history.replaceState(L,"",location.href),o(L)}},"onHashchange"),!0)},12981:(O,A,l)=>{l.d(A,{r:()=>h});function h(i,u){const d=i.createElement("template");return d.innerHTML=u,i.importNode(d.content,!0)}t(h,"parseHTML")},16074:(O,A,l)=>{l.d(A,{T:()=>d});var h=l(74395),i=l(36192),u=l(41307);function d(E,S,y){(0,i.xc)()?(0,h.visit)(E,{...y}):(0,u.ZP)({...S,url:E})}t(d,"softNavigate")},45922:(O,A,l)=>{l.d(A,{b:()=>u});var h=l(7739);let i=[];function u(o,w=!1){o.timestamp===void 0&&(o.timestamp=new Date().getTime()),o.loggedIn=y(),i.push(o),w?S():E()}t(u,"sendStats");let d=null;async function E(){await h.C,d==null&&(d=window.requestIdleCallback(S))}t(E,"scheduleSendStats");function S(){if(d=null,!i.length)return;const o=document.head?.querySelector('meta[name="browser-stats-url"]')?.content;if(!o)return;const w=JSON.stringify({stats:i});try{navigator.sendBeacon&&navigator.sendBeacon(o,w)}catch{}i=[]}t(S,"flushStats");function y(){return!!document.head?.querySelector('meta[name="user-login"]')?.content}t(y,"isLoggedIn"),document.addEventListener("pagehide",S),document.addEventListener("visibilitychange",S)}}]);})();

//# sourceMappingURL=app_assets_modules_github_soft-navigate_ts-8b1a24f6c7d2.js.map