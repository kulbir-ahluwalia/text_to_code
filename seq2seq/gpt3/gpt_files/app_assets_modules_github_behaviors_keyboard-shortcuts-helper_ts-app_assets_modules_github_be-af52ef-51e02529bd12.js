"use strict";(()=>{var ne=Object.defineProperty;var s=(M,w)=>ne(M,"name",{value:w,configurable:!0});(globalThis.webpackChunk=globalThis.webpackChunk||[]).push([["app_assets_modules_github_behaviors_keyboard-shortcuts-helper_ts-app_assets_modules_github_be-af52ef"],{7679:(M,w,l)=>{l.d(w,{H:()=>f,v:()=>p});var h=l(59753);function p(){const c=document.getElementById("ajax-error-message");c&&(c.hidden=!1)}s(p,"showGlobalError");function f(){const c=document.getElementById("ajax-error-message");c&&(c.hidden=!0)}s(f,"hideGlobalError"),(0,h.on)("deprecatedAjaxError","[data-remote]",function(c){const r=c.detail,{error:d,text:E}=r;c.currentTarget===c.target&&(d==="abort"||d==="canceled"||(/<html/.test(E)?(p(),c.stopImmediatePropagation()):setTimeout(function(){c.defaultPrevented||p()},0)))}),(0,h.on)("deprecatedAjaxSend","[data-remote]",function(){f()}),(0,h.on)("click",".js-ajax-error-dismiss",function(){f()})},5287:(M,w,l)=>{l.d(w,{Ty:()=>p,YE:()=>f,Zf:()=>c});var h=l(11793);const p=s(()=>{const r=document.querySelector("meta[name=keyboard-shortcuts-preference]");return r?r.content==="all":!0},"areCharacterKeyShortcutsEnabled"),f=s(r=>/Enter|Arrow|Escape|Meta|Control|Esc/.test(r)||r.includes("Alt")&&r.includes("Shift"),"isNonCharacterKeyShortcut"),c=s(r=>{const d=(0,h.EL)(r);return p()?!0:f(d)},"isShortcutAllowed")},72669:(M,w,l)=>{l.d(w,{L$:()=>E,nj:()=>b});var h=l(6741),p=l(59753),f=l(40987),c=l(64463),r=l(65935),d=l(56238);(0,c.N7)(".js-task-list-container .js-task-list-field",function(e){const o=e.closest(".js-task-list-container");E(o),F(o)}),(0,p.on)("task-lists-move","task-lists",function(e){const{src:o,dst:t}=e.detail,u=e.currentTarget.closest(".js-task-list-container");m(u,"reordered",{operation:"move",src:o,dst:t})}),(0,p.on)("task-lists-check","task-lists",function(e){const{position:o,checked:t}=e.detail,u=e.currentTarget.closest(".js-task-list-container");m(u,`checked:${t?1:0}`,{operation:"check",position:o,checked:t})});function E(e){if(e.querySelector(".js-task-list-field")){const o=e.querySelectorAll("task-lists");for(const t of o)if(t instanceof f.Z){t.disabled=!1;const u=t.querySelectorAll("button");for(const S of u)S.disabled=!1}}}s(E,"enableTaskList");function b(e){for(const o of e.querySelectorAll("task-lists"))if(o instanceof f.Z){o.disabled=!0;const t=o.querySelectorAll("button");for(const u of t)u.disabled=!0}}s(b,"disableTaskList");function m(e,o,t){const u=e.querySelector(".js-comment-update");b(e),F(e);const S=u.elements.namedItem("task_list_track");S instanceof Element&&S.remove();const y=u.elements.namedItem("task_list_operation");y instanceof Element&&y.remove();const A=document.createElement("input");A.setAttribute("type","hidden"),A.setAttribute("name","task_list_track"),A.setAttribute("value",o),u.appendChild(A);const k=document.createElement("input");if(k.setAttribute("type","hidden"),k.setAttribute("name","task_list_operation"),k.setAttribute("value",JSON.stringify(t)),u.appendChild(k),!u.elements.namedItem("task_list_key")){const O=u.querySelector(".js-task-list-field").getAttribute("name").split("[")[0],I=document.createElement("input");I.setAttribute("type","hidden"),I.setAttribute("name","task_list_key"),I.setAttribute("value",O),u.appendChild(I)}e.classList.remove("is-comment-stale"),(0,d.Bt)(u)}s(m,"saveTaskList"),(0,r.AC)(".js-task-list-container .js-comment-update",async function(e,o){const t=e.closest(".js-task-list-container"),u=e.elements.namedItem("task_list_track");u instanceof Element&&u.remove();const S=e.elements.namedItem("task_list_operation");S instanceof Element&&S.remove();let y;try{y=await o.json()}catch(A){let k;try{k=JSON.parse(A.response.text)}catch{}if(k&&k.stale){const C=t.querySelector(".js-task-list-field");C.classList.add("session-resumable-canceled"),C.classList.remove("js-session-resumable")}else A.response.status===422||window.location.reload()}y&&(S&&y.json.source&&(t.querySelector(".js-task-list-field").value=y.json.source),E(t),requestAnimationFrame(()=>F(t)))});let v=!1,_=!1,T=null;function i(e){e.inputType==="insertLineBreak"?v=!0:v=!1}s(i,"tryAutoCompleteOnBeforeInput");function a(e){const o=e;if(!v&&!(o.inputType==="insertLineBreak"))return;const t=o.target;n(t),v=!1}s(a,"autoCompleteOnInput");function n(e){const o=Y(e.value,[e.selectionStart,e.selectionEnd]);o!==void 0&&g(e,o)}s(n,"listAutocomplete");function g(e,o){if(T===null||T===!0){e.contentEditable="true";try{v=!1;let t;o.commandId===X.insertText?(t=o.autocompletePrefix,o.writeSelection[0]!==null&&o.writeSelection[1]!==null&&(e.selectionStart=o.writeSelection[0],e.selectionEnd=o.writeSelection[1])):e.selectionStart=o.selection[0],T=document.execCommand(o.commandId,!1,t)}catch{T=!1}e.contentEditable="false"}if(!T){try{document.execCommand("ms-beginUndoUnit")}catch{}e.value=o.text;try{document.execCommand("ms-endUndoUnit")}catch{}e.dispatchEvent(new CustomEvent("input",{bubbles:!0,cancelable:!0}))}o.selection[0]!=null&&o.selection[1]!=null&&(e.selectionStart=o.selection[0],e.selectionEnd=o.selection[1])}s(g,"updateElementText");function L(e){if(_)return;const o=e;if(o.key==="Enter"&&o.shiftKey&&!o.metaKey){const t=o.target,u=G(t.value,[t.selectionStart,t.selectionEnd]);if(u===void 0)return;g(t,u),o.preventDefault(),(0,p.f)(t,"change")}}s(L,"handleShiftEnter");function x(){_=!0}s(x,"onCompositionStart");function U(){_=!1}s(U,"onCompositionEnd");function $(e){if(_)return;const o=e;if(o.key==="Escape"){ee(e);return}if(o.key!=="Tab")return;const t=o.target,u=Q(t.value,[t.selectionStart,t.selectionEnd],o.shiftKey);u!==void 0&&(o.preventDefault(),g(t,u))}s($,"updateIndentation"),(0,c.N7)(".js-task-list-field",{subscribe:e=>(0,h.qC)((0,h.RB)(e,"keydown",$),(0,h.RB)(e,"keydown",L),(0,h.RB)(e,"beforeinput",i),(0,h.RB)(e,"input",a),(0,h.RB)(e,"compositionstart",x),(0,h.RB)(e,"compositionend",U))});var X=(e=>(e.insertText="insertText",e.delete="delete",e))(X||{});const Z=/^(\s*)?/;function G(e,o){const t=o[0];if(!t||!e)return;const u=e.substring(0,t).split(`
`),y=u[u.length-1]?.match(Z);if(!y)return;const k=`
${y[1]||""}`;return{text:e.substring(0,t)+k+e.substring(t),autocompletePrefix:k,selection:[t+k.length,t+k.length],commandId:"insertText",writeSelection:[null,null]}}s(G,"addSoftNewline");const z=/^(\s*)([*-]|(\d+)\.)\s(\[[\sx]\]\s)?/;function J(e,o){let t=e.split(`
`);return t=t.map(u=>{if(u.replace(/^\s+/,"").startsWith(`${o}.`)){const S=u.replace(`${o}`,`${o+1}`);return o+=1,S}return u}),t.join(`
`)}s(J,"updateRemainingNumberBullets");function Y(e,o){const t=o[0];if(!t||!e)return;const u=e.substring(0,t).split(`
`),S=u[u.length-2],y=S?.match(z);if(!y)return;const A=y[0],k=y[1],C=y[2],R=parseInt(y[3],10),O=Boolean(y[4]),I=!isNaN(R);let B=`${I?`${R+1}.`:C} ${O?"[ ] ":""}`,j=e.indexOf(`
`,t);j<0&&(j=e.length);const H=e.substring(t,j);if(H.startsWith(B)&&(B=""),S.replace(A,"").trim().length>0||H.trim().length>0){let P=`${k}${B}`,D=e.substring(t);const K=P.length;let W=[null,null];const te=/^\s*$/g;let V=e.substring(0,t)+P+D;return I&&!e.substring(t).match(te)&&(D=J(e.substring(t),R+1),P+=D,W=[t,t+P.length],V=e.substring(0,t)+P),{text:V,autocompletePrefix:P,selection:[t+K,t+K],commandId:"insertText",writeSelection:W}}else{const P=t-`
${A}`.length;return{autocompletePrefix:"",text:e.substring(0,P)+e.substring(t),selection:[P,P],commandId:"delete",writeSelection:[null,null]}}}s(Y,"autocompletedList");function Q(e,o,t){const u=o[0]||0,S=o[1]||u;if(o[0]===null||u===S)return;const y=e.substring(0,u).lastIndexOf(`
`)+1,A=e.indexOf(`
`,S-1),k=A>0?A:e.length-1,C=e.substring(y,k).split(`
`);let R=!1,O=0,I=0;const q=[];for(const N of C){const P=N.match(/^\s*/);if(P){let D=P[0];const K=N.substring(D.length);if(t){const W=D.length;D=D.slice(0,-2),O=R?O:D.length-W,R=!0,I+=D.length-W}else D+="  ",O=2,I+=2;q.push(D+K)}}const B=q.join(`
`),j=e.substring(0,y)+B+e.substring(k),H=[Math.max(y,u+O),S+I];return{text:j,selection:H,autocompletePrefix:B,commandId:"insertText",writeSelection:[y,k]}}s(Q,"indent");function ee(e){const t=e.target;t.selectionDirection==="backward"?t.selectionEnd=t.selectionStart:t.selectionStart=t.selectionEnd}s(ee,"deselectText");function F(e){if(document.querySelectorAll("tracked-issues-progress").length===0||e.closest(".js-timeline-item"))return;const t=e.querySelectorAll(".js-comment-body [type=checkbox]"),u=t.length,S=Array.from(t).filter(A=>A.checked).length,y=document.querySelectorAll("tracked-issues-progress[data-type=checklist]");for(const A of y)A.setAttribute("data-completed",String(S)),A.setAttribute("data-total",String(u))}s(F,"updateProgress")},12737:(M,w,l)=>{l.d(w,{W:()=>p});var h=l(59753);async function p(f){const r=document.querySelector("#site-details-dialog").content.cloneNode(!0),d=r.querySelector("details"),E=d.querySelector("details-dialog"),b=d.querySelector(".js-details-dialog-spinner");f.detailsClass&&d.classList.add(...f.detailsClass.split(" ")),f.dialogClass&&E.classList.add(...f.dialogClass.split(" ")),f.label?E.setAttribute("aria-label",f.label):f.labelledBy&&E.setAttribute("aria-labelledby",f.labelledBy),document.body.append(r);const m=await f.content;return b.remove(),E.prepend(m),d.addEventListener("toggle",()=>{d.hasAttribute("open")||((0,h.f)(E,"dialog:remove"),d.remove())}),E}s(p,"dialog")},29719:(M,w,l)=>{l.d(w,{D:()=>c,a:()=>f});var h=l(17463),p=l(12981);async function f(r,d,E){const b=new Request(d,E);b.headers.append("X-Requested-With","XMLHttpRequest");const m=await self.fetch(b);if(m.status<200||m.status>=300)throw new Error(`HTTP ${m.status}${m.statusText||""}`);return(0,h.t)((0,h.P)(r),m),(0,p.r)(r,await m.text())}s(f,"fetchSafeDocumentFragment");function c(r,d,E=1e3){return s(async function b(m){const v=new Request(r,d);v.headers.append("X-Requested-With","XMLHttpRequest");const _=await self.fetch(v);if(_.status<200||_.status>=300)throw new Error(`HTTP ${_.status}${_.statusText||""}`);if(_.status===200)return _;if(_.status===202)return await new Promise(T=>setTimeout(T,m)),b(m*1.5);throw new Error(`Unexpected ${_.status} response status from poll endpoint`)},"poll")(E)}s(c,"fetchPoll")},56238:(M,w,l)=>{l.d(w,{Bt:()=>r,DN:()=>b,KL:()=>_,Se:()=>E,qC:()=>T,sw:()=>m});var h=l(59753),p=l(2061),f=l(7679);(0,h.on)("click",".js-remote-submit-button",async function(i){const n=i.currentTarget.form;i.preventDefault();let g;try{g=await fetch(n.action,{method:n.method,body:new FormData(n),headers:{Accept:"application/json","X-Requested-With":"XMLHttpRequest"}})}catch{}g&&!g.ok&&(0,f.v)()});function c(i,a,n){return i.dispatchEvent(new CustomEvent(a,{bubbles:!0,cancelable:n}))}s(c,"fire");function r(i,a){a&&(d(i,a),(0,p.j)(a)),c(i,"submit",!0)&&i.submit()}s(r,"requestSubmit");function d(i,a){if(!(i instanceof HTMLFormElement))throw new TypeError("The specified element is not of type HTMLFormElement.");if(!(a instanceof HTMLElement))throw new TypeError("The specified element is not of type HTMLElement.");if(a.type!=="submit")throw new TypeError("The specified element is not a submit button.");if(!i||i!==a.form)throw new Error("The specified element is not owned by the form element.")}s(d,"checkButtonValidity");function E(i,a){if(typeof a=="boolean")if(i instanceof HTMLInputElement)i.checked=a;else throw new TypeError("only checkboxes can be set to boolean value");else{if(i.type==="checkbox")throw new TypeError("checkbox can't be set to string value");i.value=a}c(i,"change",!1)}s(E,"changeValue");function b(i,a){for(const n in a){const g=a[n],L=i.elements.namedItem(n);(L instanceof HTMLInputElement||L instanceof HTMLTextAreaElement)&&(L.value=g)}}s(b,"fillFormValues");function m(i){if(!(i instanceof HTMLElement))return!1;const a=i.nodeName.toLowerCase(),n=(i.getAttribute("type")||"").toLowerCase();return a==="select"||a==="textarea"||a==="input"&&n!=="submit"&&n!=="reset"||i.isContentEditable}s(m,"isFormField");function v(i){return new URLSearchParams(i)}s(v,"searchParamsFromFormData");function _(i,a){const n=new URLSearchParams(i.search),g=v(a);for(const[L,x]of g)n.append(L,x);return n.toString()}s(_,"combineGetFormSearchParams");function T(i){return v(new FormData(i)).toString()}s(T,"serialize")},55741:(M,w,l)=>{l.d(w,{M:()=>f,T:()=>c});var h=l(14840),p=l(56238);function f(n,g=!1){return c(n)||v(n,g)||i(n)||a(n)}s(f,"hasInteractions");function c(n){for(const g of n.querySelectorAll("input, textarea"))if((g instanceof HTMLInputElement||g instanceof HTMLTextAreaElement)&&r(g))return!0;return!1}s(c,"hasDirtyFields");function r(n){if(n instanceof HTMLInputElement&&(n.type==="checkbox"||n.type==="radio")){if(n.checked!==n.defaultChecked)return!0}else if(n.value!==n.defaultValue)return!0;return!1}s(r,"formFieldValueChanged");let d;async function E(n,g){d=n;try{await g()}finally{d=null}}s(E,"withActiveElement");function b(n){return d instanceof Element?d:n&&n.ownerDocument&&n.ownerDocument.activeElement?n.ownerDocument.activeElement:null}s(b,"getActiveElement");let m;document.addEventListener("mouseup",function(n){m=n.target});function v(n,g){const L=b(n);return L===null||g&&L===n?!1:L===n&&(0,p.sw)(L)||n.contains(L)&&!T(L)?!0:m instanceof Element&&n.contains(m)&&!!m.closest("details[open] > summary")}s(v,"hasFocus");const _="a[href], button";function T(n){if(n instanceof h.Z)return!0;const g=n instanceof HTMLAnchorElement||n instanceof HTMLButtonElement,L=n.parentElement?.classList.contains("task-list-item");if(g&&L)return!0;if(!(m instanceof Element))return!1;const x=n.closest(_);if(!x)return!1;const U=m.closest(_);return x===U}s(T,"activeElementIsSafe");function i(n){return n.matches(":active:enabled")}s(i,"hasMousedown");function a(n){return!!(n.closest(".is-dirty")||n.querySelector(".is-dirty"))}s(a,"markedAsDirty")},17463:(M,w,l)=>{l.d(w,{P:()=>h,t:()=>f});function h(c){const r=[...c.querySelectorAll("meta[name=html-safe-nonce]")].map(d=>d.content);if(r.length<1)throw new Error("could not find html-safe-nonce on document");return r}s(h,"getDocumentHtmlSafeNonces");class p extends Error{constructor(r,d){super(`${r} for HTTP ${d.status}`);this.response=d}}s(p,"ResponseError");function f(c,r,d=!1){const E=r.headers.get("content-type")||"";if(!d&&!E.startsWith("text/html"))throw new p(`expected response with text/html, but was ${E}`,r);if(d&&!(E.startsWith("text/html")||E.startsWith("application/json")))throw new p(`expected response with text/html or application/json, but was ${E}`,r);const b=r.headers.get("x-html-safe");if(b){if(!c.includes(b))throw new p("response X-HTML-Safe nonce did not match",r)}else throw new p("missing X-HTML-Safe nonce",r)}s(f,"verifyResponseHtmlSafeNonce")},45075:(M,w,l)=>{l.d(w,{ZG:()=>r,q6:()=>E,w4:()=>d});var h=l(8439);let p=!1;const f=new h.Z;function c(b){const m=b.target;if(m instanceof HTMLElement&&m.nodeType!==Node.DOCUMENT_NODE)for(const v of f.matches(m))v.data.call(null,m)}s(c,"handleFocus");function r(b,m){p||(p=!0,document.addEventListener("focus",c,!0)),f.add(b,m),document.activeElement instanceof HTMLElement&&document.activeElement.matches(b)&&m(document.activeElement)}s(r,"onFocus");function d(b,m,v){function _(T){const i=T.currentTarget;!i||(i.removeEventListener(b,v),i.removeEventListener("blur",_))}s(_,"blurHandler"),r(m,function(T){T.addEventListener(b,v),T.addEventListener("blur",_)})}s(d,"onKey");function E(b,m){function v(_){const{currentTarget:T}=_;!T||(T.removeEventListener("input",m),T.removeEventListener("blur",v))}s(v,"blurHandler"),r(b,function(_){_.addEventListener("input",m),_.addEventListener("blur",v)})}s(E,"onInput")},2061:(M,w,l)=>{l.d(w,{j:()=>h,u:()=>p});function h(f){const c=f.closest("form");if(!(c instanceof HTMLFormElement))return;let r=p(c);if(f.name){const d=f.matches("input[type=submit]")?"Submit":"",E=f.value||d;r||(r=document.createElement("input"),r.type="hidden",r.classList.add("js-submit-button-value"),c.prepend(r)),r.name=f.name,r.value=E}else r&&r.remove()}s(h,"persistSubmitButtonValue");function p(f){const c=f.querySelector("input.js-submit-button-value");return c instanceof HTMLInputElement?c:null}s(p,"findPersistedSubmitButtonValue")},25833:(M,w,l)=>{l.d(w,{Z:()=>T});var h=l(12737),p=l(29719),f=l(64463);function c(i){return new Promise(a=>{i.addEventListener("dialog:remove",a,{once:!0})})}s(c,"waitForDialogClose");function r(i){const a=document.querySelector(".sso-modal");!a||(a.classList.remove("success","error"),i?a.classList.add("success"):a.classList.add("error"))}s(r,"setModalStatus");function d(i){const a=document.querySelector("meta[name=sso-expires-around]");a&&a.setAttribute("content",i)}s(d,"updateExpiresAroundTag");async function E(){const i=document.querySelector("link[rel=sso-modal]"),a=await(0,h.W)({content:(0,p.a)(document,i.href),dialogClass:"sso-modal"});let n=null;const g=window.external;if(g.ssoComplete=function(L){L.error?(n=!1,r(n)):(n=!0,r(n),d(L.expiresAround),window.focus()),g.ssoComplete=null},await c(a),!n)throw new Error("sso prompt canceled")}s(E,"ssoPrompt"),(0,f.N7)(".js-sso-modal-complete",function(i){if(window.opener&&window.opener.external.ssoComplete){const a=i.getAttribute("data-error"),n=i.getAttribute("data-expires-around");window.opener.external.ssoComplete({error:a,expiresAround:n}),window.close()}else{const a=i.getAttribute("data-fallback-url");a&&(window.location.href=a)}});function b(i){if(!(i instanceof HTMLMetaElement))return!0;const a=parseInt(i.content);return new Date().getTime()/1e3>a}s(b,"expiresSoon");async function m(){const i=document.querySelector("link[rel=sso-session]"),a=document.querySelector("meta[name=sso-expires-around]");if(!(i instanceof HTMLLinkElement)||!b(a))return!0;const n=i.href;return await(await fetch(n,{headers:{Accept:"application/json","X-Requested-With":"XMLHttpRequest"}})).json()}s(m,"fetchSsoStatus");let v=null;function _(){v=null}s(_,"clearActiveSsoPrompt");async function T(){await m()||(v||(v=E().then(_).catch(_)),await v)}s(T,"__WEBPACK_DEFAULT_EXPORT__")},6741:(M,w,l)=>{l.d(w,{RB:()=>p,qC:()=>f,w0:()=>h});class h{constructor(r){this.closed=!1,this.unsubscribe=()=>{r(),this.closed=!0}}}s(h,"Subscription");function p(c,r,d,E={capture:!1}){return c.addEventListener(r,d,E),new h(()=>{c.removeEventListener(r,d,E)})}s(p,"fromEvent");function f(...c){return new h(()=>{for(const r of c)r.unsubscribe()})}s(f,"compose")}}]);})();

//# sourceMappingURL=app_assets_modules_github_behaviors_keyboard-shortcuts-helper_ts-app_assets_modules_github_be-af52ef-4aa96fc99887.js.map