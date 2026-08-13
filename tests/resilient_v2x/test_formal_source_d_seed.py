from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import importlib.util
import json
import os
import stat
import struct
import sys
import textwrap
import zipfile
import zlib
from pathlib import Path
from types import SimpleNamespace

import pytest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "tools/resilient_v2x/formal_source_d_seed.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("formal_source_d_seed", MODULE_PATH)
    assert spec is not None and spec.loader is not None
    loaded = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = loaded
    spec.loader.exec_module(loaded)
    return loaded


module = _load_module()


_LIVE_SOURCE_C_B85 = (
    "c-ri}YjfK;vM~JJzk<uHdLzxqimsOAT|ZTdtt2|*TiH%_a<VQKMMAbVvZy5`CmtXF_oo{-0w"
    "73HwzGTAd7tc#C6Tx^8jVJy(U(8}>Fqj+-%e(Ww`6heEFV_4(W1TMy6%a;BJs?h&;B5ezX+X"
    "s_%(fSR=31Sh(9Nxb8+>%(`^hIPP|^MW_QGykZBYXXBGRi#cXj?hgNoWrg3!Vc;0lqTF1omo"
    "Y~znidWF&B3k*YS+q!YR`J8WL(BQ|Ik|}=e<q4u`N_8(@*^P2l{2P~MsXa)ym&Atetb9A3Rg"
    "UD;oo6Te>!kn&$|O)JkPB<QA`We0k&18y0fz*Ac=35ew+~T`_@lx=d+3U$v)sgeZ88^#nXQz"
    "(L#KW67emTjU=~f>0~`w#!)~LSu%N$UsrxSoy}!U#3V!7(qADUVsz*5v*w%w7SE$(_5**ia;"
    "t|WNkplWC|u9U6+DtJVJhTXe~CY9&MBZMyjm~kq~=_bf3C?Q;1$;K90yBLBC4$y|6E5aC<-r"
    "=<$SgR?7bUaj=b^7@WaR(o*eCUyp!SS`2Fbe$~zi=fDh-usq8rL&rNRqR(EedoCMweUZdTeP"
    "N$RBBxtl7q~GYZ292=OCjGEO!fB^NNPn+0^}9j8-3ll0J8TRF)1cGrx;s_1)5EjV_u~(Fowf"
    "VT7U{Mc0rasqne4Ti{a%02YqWcfX}1+j{J{6ytwF0fnKnpg((=ik-)gk_(D8K8ola<HpT@^W"
    "-qrB(H*b6dJ-CBLw;MFVcF+mKPPY;C{a!Z=Ll{ld@9s@Pcjt6?HU2#ESk23`PZx(H?`Q~3jI"
    "MYalVB1yyFTfWPPan_{eE{6bXwhp-wsKy*CF;ch8Kq)@!{$4WQ486B$>@;WU<=4Z~fR!qIDd"
    "Q-C(-cB-3`#?X+8UpsM;G$*Q=ycVDhXm(X6b*=g<dx`kygKMn!0DWWH>1_^>`yU`!CJH#jb7"
    "N8RT57X9|_<k$s4Z2~sN2dNB>2{~2*KT$j&9K`b-LTi{4-mt=zr(YWbNKe|c;uabx%zl^>Yb"
    "e)fAJ1K4o^Q&_;{B=w3yCrlDE_8f~?-MU*5z|$XyhYxpz-)y=XCi@Pg<L2>TAuL&Ey<!L605"
    "9!;mS02rRv4(5K6Y-&YLfd^CYmbT-)r7e{$%-QC~;#hh<TabiT+@gZNT20r<HeHaLh$3-|D$"
    "^i}mL%SyMmS5L2w81W<u>}>f*HBnqRJe0FxYnz*kiV+b>F(*4zlmJWPzY<S0yIph7<a;*u#B"
    "p;jM@t+?K#It)e|DR|VG340`gGF`--d##*pD#L+!nBFWoiy~MqOb{ww#`F6DtFMjMTA`fT-c"
    "Fs8!4W8A;cjABZro^Z6#`9Y<K=GE?>^%0`q_bT^eA+yeJG|EzXPE)6NU|~;!Dk*9nY?DZ-Sa"
    "}SB#RLC{s-^7zcPqxV-rgsRz0xE#VU?KZriGEvYxHTHWh?W`ON;lNAb7mJo;X$4z?bwJi1SK"
    "V%3(bd3P7dh3ef#(Kj!7Sb)sDng!2j07-_5;N_=RZy>wH6x5`ISF$LZf>2@yNNEK)i{fWDaO"
    "bb$*^g(|F4Zdo5!W)g_vh=<I2F5<(s7w;VufaQpxhc_kK3{%WuM!!Ca{!v?t{QIpOq;N8v`5"
    "*V6Od@zfJk=7ZO&49cejRZc`<Ovk#l#Fba}#m6S)`9qfDM)@dBU@eWS7C5k4n5b;P>rYw^^w"
    "wy#Ee{5fk%OE)wqiCLyVAivFD5xgcP<V@-rl2+|7`#aotrFM)mRnUqV7IAIw4ZINyJ34OsBV"
    "f{*0LLSBugdE<j6*TX{oz;<rY*j*+c9vlKUC5r?>bjfy0-_q#<uZF@1U3R+6O>7Ukb_N~Y}p"
    "EnD)&NGy!NUR$pv_m`ZC^l+HGExpHQtKzUIS7pj1*ln|qHnbqN=}l9RDX&lw4%S8!Mb-1pBZ"
    "ZZ)qx<LwnJ;qZlGi~|tL&Iq*%wTM{a6Q0EZ3{#IaRl+6!`1JpYttpn~GA7DK~d|<*XF5sfLt"
    "hp3|Kw(>$lXWrN;Sg>Syk=}m6I&#Wvq;&RpLitu1Bg`cGtvQddK_BpxnVMku#0WuaZbJ~f?4"
    "IDMa>#D9R9G{+lx=K}qk6af^R%^Ot@1yRtv)5{qUO4Erx*b38d+n~@>-&2_tI-I9y~d!~?}b"
    "f2Y<3&npfiDD{chLy_gc;FbTDZb3~fU0y(R2N8YZC2_Jp+jKK$(yzulkgbvw<TAGEuZPT21C"
    "rh{fDXooEl2918N*X{e0UKmWt6y~By27Zf_=~8^*UG^H2R%;M;8_h{S91NNvX|x;RV6Qdk4j"
    "R2dr-cg9cC+R8TK=F1#hM+z+v*3?MzcR?4yI+gOyVHzHdrpP-;UoS{k^c$1X-lno9s<m0cmv"
    "l&A^|8QyBlC-<$R)o$0jQ8o(qp$z&Q#CxmpHVSnOxgKhiuf_3QEgEjWu-5d1VVL0iLaN2700"
    "L6Z{-v=Pt{vM2MFzNJLeZLp@K56^GVA2ShohfNf8lCQRF9;`_h}N9e(D$2r(`K{Z343kmp)u"
    "|DI-O}V0P^ZLf=Lg0XmttvKWKIbVXxPobeln=*YrE2)ol73`%40U1%wU*oh=wYhhAI#4))ZY"
    "gfOX--XLg(?JjAAfcee@zO({=Fl`R@fLi-x(wes0LAwF8)D1g9ztb6P?iJ9h_v{7BwV2=@5D"
    "&EIx7&kue;SZ>tJ%ZZ^{10g8wS-0r;{lRu}7MNL8mckHk<7!{3ZR~-hgx(&+0y%<6sa%uhXE"
    "}>-u5PYPF}n-|hgZ^?LwNXF7q`J(vZ*(dx7a={Fl4AE+M|U@Mrm29r|V({Ew}=Hm|c_DEwg2"
    "-|)iI7QeTwD&sAPKPx7paBbZFbF2oy;cyw{P$sv1%qkJ_ouxH%u6F|G=l9rzQu0$2EDx&nKp"
    "X;sSgZ_wA=n<&<_08pfw$|!!~KP2Y#p9YBl>oztshX(QQnF!0&gU=SHKwb+_tVcZuI?OriOp"
    "16&F^4O;NapG-Ss3SWB@U@FtF0}HIvYX|*7KWy|SlR*bw?(Ma@-Dh;3&NMK<!JzB6C+&9A?>"
    "0#YEN$AF0;+p{1D3E4E0+xRf@wc!PMUiizXi;jG)bq~B8?#2+t@kpnDBK4w9@N#yQDJ-ntR>"
    "7=AZ>s-tt4hYCrJzdOgzX?g8s+gs`?-gHEU2@_~B*O9NikZFiu*r*tnmJvzo4tE=(*;o%j%C"
    "|muY=d%S2^q#<(a1qXl%dht2jis#yJL}&0i`kUmJ=*#B+<Sj98vUoegXL`LO=Cj-fYwL9pN|"
    "f&Mn|$B-eFDS=nt}hEs<Sju`Lq+*@3Ug_c^ep1uT%Umrt-wOIWVXE)G8;#I1U>QE&VjtedUA"
    "YrOXkKOGIRP_tE6Uw%0M<Q<-UI>jM((#MCx^WnSk@%ZWsUeYyNHK$SC`LlD$>;tH6z;eD$oW"
    "NiD!EE)gdq>ayp)+hY8gD<tAIG0C-L-=nT98cYJHzASv%iTxMF-=Nj>AXZTg|ET`DYJ5pOg`"
    "w4-bDEvI|5mklft`_mr8}n{~+&@9xOr281qr(Lmhv(l>Avp^`+srIv@M?GT?g>uvh{Y3K0M<"
    "<;4VcXsZboE?2SR&Onpl~89Zp-$HT`|Hj92NGLWjVLR(db#(KA1f+>{q5}Hw@Vl#-Uy>LK*f"
    "EFNK63N<gI=~j8n18mBDUx=kw_Fvv+ZJc9qtQmMcpgSKNZr3t6Z&zBoHQ8J%8vzm2{WQnh<{"
    "=+=~9zyGfNJRe?t)E)rcc^}VCM&?7K;`=kalg1b7)8p~Gi{Zr=&usN%_`CNXXYVexXD5fBy}"
    "ymGKEh;aFVY1-@4UU^@#U49E+i)0Vi#AVi&GeSOzL<=Je!kB?0#+R4*cEe_2XXW36OSpJQ`k"
    "{91G)tjkVe>QR~#Z8VwIWjxLapM#u8T|3b{o=hpAk?q%WdqpQ1f1dxu8kAdVbuRa~&>=e}H7"
    "e=rMlhrzehfx31#SkywS*=Ux{lr{u?(J%|bZZX%Oz=z6#=kf8yZ^^JCa${n-j87oc$cG#&oG"
    "jYXVgN&ZQ}B3)Z3jpNN;XV;q$W#EZlGOn>*TzPZwi^i9)dd_U&VSUr+mw`E5V3jxWZPjFCb{"
    "=Vyl>alGBe&c)U5l$~E)49BP5yWtfO!sQq^E7U-a1f^QC(CG8<*j}cyb9i=ob#X=~1&vmY$E"
    "PD&{?icXEMrW8JKS_HY_$3yIqrdY+-`$h(;2kejXoZgKv?d;?gyuqy$+P^2ED<=@AUWff?gM"
    "dgb<A>Y&{;2J`4}PcppbY*jp|y3m|AunjmGg8f}mkr%eK|GzP5>P#A2qdz}WHTHr)90cocn!"
    "il2OpHB8#AeV(*5cWa%XyidqqU&C(0Y|+_t2>wko&KQR9ZV;Kpf&0Bf@W*b==MO^n+`&JN<z"
    "|}w&6I_B3+PzLHdWY)`r2KUz}Z?9iAN*KtQIgK8O?{!7~Ch-0Zf{<Ormly|CSC3<d)fr9tKa"
    "5v&bIr~s&Q0%xtTLjsUXdU+6(=sIk*+oaL$Ok0yZXf)__;CunHPaDMd{vd3Em_O)+eK@+pIS"
    "~$y4JZrcn;n$W!uB5RdU$$vIz9wqq`G$5E3-H{8jde^4~L&FhsV44e)qoVnpG}`C+Ei_xy9o"
    "%v&e@FcnM;{#V25$vLL)_m?fpCf+)d7mYMKo*AMolJ-^v$;YqF0ZuJNW8++YG0B8OQoXCU8p"
    "brO|e!BrOCh2#&AZ)k$FcW*7ZfF_v87%MP;TLawdNleyzhgWYbo<?KGHC#1HK%<b;(qAw`9S"
    "%7zuV~S!D+7F>iF%j+YP48E}+}@12`!YI95z89dpc$k1jL4`TY>6sxyW42@~0Z$s<5Ia8mBH"
    "dectdZv-%b1JZ+mhG;A0qZY*P`<)h?bzr^G-o!@2`4K0q4F4d0zdvZ~1p^<5a}su&trk$$v<"
    "C>>BVhn56l7|DuQed3X$Yr#uy)||)&j0F-D@|qU7d`sF2;wK+%_xIA==P&gKneWXuvsau-68"
    "LwMTjZ=>lJZHQVn^;g~jUPX;{#)7&2bUg0d-YwjTsemC18c62;KrmU`@%yMbNXaYF&bpv1wy"
    "#eXB`>ht7Q6`gC4|?o}$Q)p`c2VyWwkP3UA5Jt=psWFzPJ0{r-7spHsV)!-30lC3`@jo2a1x"
    "vB4F=6Aoc^K5L8}qMiLo636?dkMY1jhP`0eHdzO<X2Z13;ipN^oPE3|0LOe!4ar+YmhQ#c+G"
    "put`UgcA^d0w-|NAVD+ewnD)06qZXD=m}2TQ}_s{z!w5$!)WX6?EU-k;h1h+Iam(bZQzDY9~"
    "x>+T5UM;!6~ng`jZ|k!A`5$Bke&G$g2r!iNJZc3p3o9!e6on6Vuz+^9DS3;Beas;5a{pW!Qp"
    "Qo(4cS@D*5Js{teL1CIM}{2p}MAs`p+J14zHXF3H=*=x14{Y#!mrJG^FN`j;HUI$0r1zMU;1"
    "EA+#yVu(b`+Y#XKOlqNpaq-49^kmMhx){6FxUe&vDax1NON=7nI0R!c2Kzkoxr#!O~2LbPuf"
    "!-Hk@WFoOYXy4xF(&lipqfI_^z+-3GO#Y=@I+AJ1^bYj?~}XuzU?#>GjB3F?L1W#1{m5o(rp"
    "idOPIAN7V>*1l-Ps;$Rjy3X&j3{~d$x`C$hyR}n<=;$eDHZMl!$8gv~45g>bv<ugD@gwjT;V"
    "kr5B!MzX6t75F!Be3Z&f<D}Gmj<}HwpZ?4@Ye@Vt1=m+Qe+?%*mpXR<EM{d~;v6AN#X}IG_F"
    "bnotw`v}gP`b;*w<37|KJ{P2U-{K1J9#8G>4){BtD&ZB_lspd>0c;-AB9fO1}nysQILhZH2$"
    "gXugW-g7Qkyc2cVG*sA9#RC@dev}%_WC+fqL@B(W=y@I!uJb@Osl{r0qfM!iJ|Hoz<Hygl|f"
    "$&S>S8`8jHA|n_s3*7Gd&zhNpa6pIA2S@#{Xc_^Z;vV87YDuGVKsIJ=>K6Y125ZyB_noQ$YO"
    "h59liEB|f@WWh=~^XSI?i3Ink0rO;`IoD`r8ye(nAhYOrr*ZT>*>~n}!2TM}g4Ne#71x|-@("
    "%#>8s^FQdc7maf?gOe)^`&EMeqmG2!RJAhW^tI1ck=L54{!nv8vEpy4m99z+JDVyM4D>ryd#"
    "f8o5$U*9zc0UL7<!;QQ8vv>n{MG#Ubs#R?a}X5eVWKRc2l?)(Rb+|5?be_WoO9@iY-Nuc<g("
    "BHhz8oKI0F9Bd;;o}Um=Lv5NX*pW2oCWz#NBtLvIji7vpqlSqNbn-l@q;)@66&$^{WgL=mj1"
    "&$^20i9hZTxp7`Q0v*#8lG#ECqbP7{K|Wa|$I<ObS5pj4})f|dV%JA*!Wt-tc}DLGqT)77Up"
    "_LUX6h9yTed6nwEMzOqwUtz86D13uI{yztqj`80Frs$h|g8o+8YaoCcP+th6<7la;cr*rE<6"
    "WiepSx!nXryb_@e_wohb_Io!lljD<!o{9&u1Yc3|JhGoZ6n8im}M6j=vJ)e`KFe`YNw7cQFc"
    "yiZo>PGax`%D+|ChMFXyY>IQJ9?jl<3GOA<%a?;WPwSZ-E%|yy#;@>y&u<xLW5uuC%e%AO;X"
    "d57`&sO9PRyx8DFY9!jbN;Ry7AT!PsiWpS>tj?{LET@Xu^ylIByB~m2zaY8Mf9elPBYF3m~u"
    "QrJM~N8&Xv^7jH484oguTkyIZgP2`q>^f;<qKrt5SWC97TZ*pp6#4zm=eu7ubCZc6*W+}J#R"
    "n=Qih7c*g|zn%YbPGLWwk7M%B^$buWY<>2fNfgZi<ioRzk*6pFFV9DZsc2;kCFPx_r-RB@6+"
    "?aX@r(Q?@ZVLuCfXj93*B-x<z5CV0OFcuGf<*r@-}VhC9uFsA+G@mw)0##&>D!uUV;v%!ApV"
    "B8har`>%|Nf`Q0ml#9MnVOp6GJZ$9%U(8ucGr7-EOy#S_=pouy#BOh%YUjZz=!Iwa$_};t*T"
    "vk_}?Vw^}lP4}*J$~W(Qs{D;-405Znsd%pcw;0i-QYX&=^Cw^Q!R)l<o+48a4nE{LAI_6-7("
    "X{Z7Wi%x}r+eo#D~Z_-cH1Iy`>H&SUo;d!7AflV{qCR)^|6_d4K~a?dM8mBQn7>&a$ruiIAR"
    "F1g!QW7etL))I!P&#1?3Qn#(C818OcRoEwQTTd7oZ&^>-j8!sI>2L|f{h00*w^6v6LW?|aGq"
    "hOUu`mF;NcyW&z&0{s?AgW9=mOeWq1;_bV60S?Lyr^q3l9?3el&>6v{Kl>yD!23;ZfG!XH|>"
    "Zc-j#L5J#i)(J3`&NXuAkPRfCX2OnC}JPq)u66}Kh75l-X3DwO>l7yblp(q-voQ$qMo*n(1{"
    "tTi;Q|-|uweaG2&Fl5iTnLt<&Qn!dKcMl$`1ERY0blScI5nP-cm9X(KX_;7NVkeT)cq`3)37"
    "pe-Gc?cB;J8E!K!77U~rRz&+@mW1bp=-e((*|ob`sSO^qx81&6<#pN;XW4ZM&{{q=lBop+Nu"
    "P)P8N+QZi2KN7=9{GcgK!E(#q`9HjGge*OHuu2ZP6o|vKlasU5zGr+2D|&c5{?8#oK<}(CxK"
    "Uwp>&GPA#a(Ha+WPErV%^o<RM=I1VYBCF$K%5<Mxz@mU5wt3e^-sz=Ce@o5RlI5KI$tok0+D"
    "phj&Z-u%04HxZc7R{txaYRRSr)Pr&x54G(7}G7?CE^N#1Q!`X@}^~?33T6C6ZJA*$SD}@`y?"
    "KOKj1x`PmjBv5EyI#N9^g8`+PC*%&rPKCWojv%klPURecmb=L_u1L&=M<2AHob0R&l|M!T9S"
    "RX_Pj=`?=?CNv*i0vmxwgpXR|Zd8J(P8eS!C<muDB=$?*JqeEK21;^eWE>MSZxW#YQk`Yrhp"
    "&TdG;f-u1na=c}f%&2`XuCW^uSE#~^DsrmM?*9S4w_@res^~>EB|Kq;<n0xI_16QZVcsE#2v"
    "7B4e6z;=AHyRtO`gkCYj^8p<xHsli2}k^XE9l=;{|V80L#YW`MfO?R`}c(FzGgf_33&(rza;"
    "4x-(0tl=6IdeP<qhCvgB<40g5)z3HIS;u6i+Xk4LSX8Jg%FBLSSYCNY@B_v4@&z86%4m|X>_"
    "PnZ6v+jpsiXeAaL@45JnY%NPrG#28JfI?4iQ0Y~+``=K;-2`D`jdwhL9<1=wPa}T1r0OGkS!"
    "a8vQ5K6`}tgqF?07_X72uq1J>Jpz420G`(0sd|B4OYEZ1`6GpriiMzeq<2bE6E=~rtG-@*?%"
    "`OTzKKg{^bzfskhvN=te?)r08D@-48-GJp`8hHv`i#HUZlwp?Pk$Z9;i>=Ijo%2|j1)GiGZ4"
    "4al%;(bm)lxvYjud!SP*XT@AVKYtWfa^N5w2*o+bCX5TzE<aG|?s+zZb7?hK#MNsx(97xpNn"
    "<p>|<Kf3uK~dG<<6+>-gyq_B9s&@C;P&qaEHlJ*LlB}ngtUa$n~IL1f%{g``*!3a){9R%$!j"
    "z3AT%pC}xiL*{v%_yFMIPK3Jz6cx<QpYu?yBt=;q+pkny%i1@OF7YZdLu-~p-RQh*?P5HuTI"
    "cm;x7nr=u94fN!&#%Vvdwg=vz*$=y^hili;PZfJsEFSSeUcUN}zb&S4p-OJ^~EK+gDx9sqqY"
    "pp+Cryjq078QkZ<lQtvSZF)g$WxE-;x%~wfJj5Ly0~5#+p~C~mbSe<*sLN;zn1N?R-*>0}3`"
    "krf90g{d9x4uCf}<E%9&8L7NQyQJXl2bhjKbh><e<{bnXMABrUdYG25U~Y!5EWCS-2u;kOUu"
    "181DGo)f@x}*UP61w0D3*x-x-)I*fVBgR4H1!UT`XTS+d9WVn)pOqY>#1`&;Dl!s)Ot`qlJK"
    "y4uVVgNYlD{J*R6rbhc{{o5^!+9pfFG0E?ETBi;yqeRxy7gD=U;vc^LK!|BFtmMug3C1h>|%"
    "W@pwbW^yQ!Xa+kMPUTX94yj)+x588^b1kt!Tj@_IJ7&siguW}GSA2^CV^ZgJQk+pthFJ>ud3"
    "C{vW#;8b|z^CA0b>n0zwTdW$~9`4XJ&9VVxPw}xh3k$mP*y8ol9zD)u7+#>CWIsIzqd^Nb%V"
    "J>tQq#>)YCH6BjrJw%B#jf`%}`@;LstH36<4sqnlp7DvC@;r3vf_bB-*0#q6Dv?fXyZ$zh<T"
    "=Z-S}_&<?4>VW~+lYT5RszQtcWRJdJILJg@<lOfnOvsPj)SLvpQC19sZiLF#!VZ2z@CX|avY"
    "{X)-kOn|(Y2%)p?#|f_*)>dDVXR^nv-%oFTJQaNLPX&l-%myR!<)<1|7ezGySr>Of^}JSKke"
    "4;E%^Zha;xRR5VO1`9D-4m1!N;Rx>c17%1^e0NSQyJ&J4!<7&T_AHPGJ!S}V@i3F`QadFFG8"
    "DuoP_5ZorI<naT#?o;@YmZvi^rxt+~mnn)wSTc`lb6goSPI;`*I$H{v<&2G1RdYUM!HPl=@r"
    "VX+004faFY54MwybP6YXsEM4(c&k&e8bvjrXSJyxDzIeUj6SrjO)qxq9GJ%(Y)M>b~$NM_B8"
    "8(CYh?dngatCD)&*w$WM($D&o3B_5nnqxpKJna5@ZWY--kgbRa&`N8n}RH>+d2av$FQq(rT?"
    "Sk4doVii@U&awOmGYbS=k3;C9J)c_)BsUTW+8}$(9sUFoE5T}z@9$Yu2KM12OitcDV<4J%|<"
    "edS?;Zw<}Hx{wbs?zUfxz_YejBrxvk~ybk^qNu9w?h(MG8IJ76#R$<5YdcDK|kQR@PWGlU=>"
    "rXx}Be^Uk7zUlOVduhf5T5gcikk#}pk(*_<vWlBp;af3P8%jx$ptx8DZQ1g=`-#Dm%2|io6_"
    "4q<v&yli>qohh!j`qf=Eww#t*U0l><y?ZwaJ^)Y}xF$#CsvSHV6f;I-&yk<yD4MQqb8jCgmi"
    "b&q6JQGOUgxC_4mO4|!|@xW;A-TN<odTP(A9i`S`5&!G>QpU;F6?x57sjiD8#zyo>BVQp5ZX"
    "|``xT$7ZTAbkk7Inkl9m7tr7ut`t^UPW@wt;U<(wi;ECdQ3ek%^`4-ei{R30#YJ0u5YtSG;y"
    "X&HZ40Q&F>U#vY8X;&K4VN{^q)QV=b|%mfcVlf6Q%Ntn6IRiRi6<A{HsryJhupZVnik178tu"
    "v!DA)gKuFT*dp^AW2$^~3Cy{H%Nf8=;5R>k*WCJv0-^!)mh1xCN?|jAEy+VN@^3ElklG*}=x"
    "0oS?yXR|!T65Q>x$bmfg^E|kWa>n7@y$)|5GpctC^#RN=Q+x(Za@kpakcix(tJlDK`HUKS@S"
    "!xyw}iO6S?blvoLwaUex0V<{t;h;g#k6`IN3R$8D_@W&h4zE6P$-^t(UHXtPbP-$R9D1^2H*"
    "uHb?mokcem3M_p{F!?<-?w+NixAT};<78NAk4A6K(;kYqB*^7Vkr_!S~qAg74z1TY_s%GluJ"
    "Au15ivreXg^}W&7UhHjdUex7rPrWgx0VlrIexNJ82et@4gm!Y=RRASxs8R5#uRWbwpS*yiOW"
    "rWFmhnb{H2n>A2ic?C5R)^zrRhNzHm(QDUq$vj&7cyhSyQ+CqQPRc9utkVj!y1gRr1udC6yc"
    "aZOQO@PGS^DnCEs});nynU903c3GW_amJB}-p5vr^|uI&e4|5pkE8uZGH&h1XDbR;!Z0D4;0"
    "_UPa5GQ5U^d$kPPyONO1-RV!4V&kx0E#iF9ID;cs2%jzApzmt2irwB6U#EP8K2(mS%t&Z)_c"
    "!d^iMObrf61Qoymm@s0WdpM8a<=f&>*Xxd7XvQX7qa_o&c0CoTG|3$&K-4$dumH#FP|=XKW("
    "!YlfDXZbUzExU?qZeP6Gkwo@GklK35TfAZ(^xYn$6su*G1q)IIZdM^Bt&8x?FlU&3~fPg_jT"
    "T2?tpK-;)k6ilaW(x{xD_V&7pD}%{oRZk^W<~FO-&{FMer)8{e7Y~f@36I7Gs~$NE4tMGXuK"
    "N1W@M!z=W?OghVSY?mKPMg#3Pl7ONi72ATXExwTg8A#P@VX9`<i?m+^!ejaKb=1j4N~hZW8+"
    "Yyd;gURp~pwIL$_@!#}Du2fe9uie>QB*GoK{RcM=<Z3BOA@xWpRs2*NlS9U}&J2qs^(n*+da"
    "m;X|C+!XOHrYoHEgpmi@WXCPDe|z%{nTj-urq@L>BAieT;B>W^OcBiXbXL_p8GL1M&Om{HU7"
    "Tyh;52MOEmi40qX`<WHi4g6|{v2R*LHgKPwAZC=Om55PnaeB1qd&WV}8MDpRS}F>U1R4|?P^"
    "&{_l)wLPHq?q&()%2^;(imqzy^9kM9fw1Ar=|dXZt!^_`G%^VXX4YWY(kk5CBaWh#vchyjnw"
    "ZR4O>DJhf{d#sHyM?PmX(exOX@OWUJ6S6AHW!}x5_W)>zJHJ$?OL;Y|%ie)|VK2tEw|Q1H1#"
    "{cP-0h_=e#v$af4<NR!_^m`o?TZpe>WvcgDUJj`U_LV6Fw9nlJ6He-Te8qml!axkUm_ZSbRv"
    "{8J@y3eep`rS7Gv4V*nL4s6PQNhHA9*gotkyhcc1o6Q>NbfODME%N-$BSuH<DvNw1ZwcDvY="
    ">^kocaUr6r+33a>LPMc5!|bH@6rYxYN3Ga{iMqi^WE;uE)Ccagx-r-dtIy-55i5yna)xEV*K"
    "0takJ!hq8U&M^x>4&kuVp8SM*JN>9gfQ@n}c?xtC0Tiy6^BD$GG+@dDM>L8ywiUfkjadq$&J"
    "xBh5m)jN5=eE6om+s0l$$W+AuO4hKc}2Dq4s;yb}9?GAzT1S-Owm?3K=QuIi1O>xg}6DPTq("
    "bB9XONmur6&Rn)d^8w(fYSY*U><d%4dWs$>ai6nu)1e|DVBp)MY;6uJj*@&c+12l_SWzcg4L"
    "@(4A&_4O%%dSUb)hX}sa?8%nltR^<<%}Uk$x4-;1bd(Y6o;zjxbgnYA54~Fo~ZjuR7FGt9yi"
    "&WIt-IPQ)1Uj#)QF5HGFfruH`W=@md*8O^&2m%?RRr)?m4EC}fH2oou<9AYNtw&HC3xJULIS"
    "MuiQtW;CEneJQNg8UG_$h1*W~><;6CIDcT|%+Qa+ou5ny96Upk#>+ieeTj-BT3o_a|4$ek1|"
    "($#VR@l+g;KSLQZ*njZNn;@+es0>MU#IVu+q$){J@R>%TAPbg2*eS@vvu+p|#1^Asm0XoIiL"
    "%Qd-05Hs;xq6Av*7g;~JA9KkN7AlG%zQGTS?FWl3PHE4L@$s1g$mYty@3;J+a7`h~l4ytrYg"
    "LJ0?U9R*mQLWZ}TQX;N>L5UO2vq<R@<jTo#KD5c(3ZhZIYRhZAw6yI69#Kr?A>xCa=*$O@`l"
    "Y#!NlwBV>v@(`&r1b?=W(a1G8^rjBAJ+J7v5+KPGX`0o1+*;YHd58>e*9f*z8D1(IcyJSQHf"
    "wHZ}m8WEcGSm5-Oyj>!OI@3E<j1$DB6d{=_F?ofh4S2S&4<zHh!leQ2@UmnP9#iCPIkFoSNg"
    "7-_$ax)Zc1bb(x^q^JU=?9R*dn?!nT}v9`<c-2j~d)hnF}5cG9X8gQk{p)*2147+mGg94Fi_"
    "b*lQcpJ7~8Kyq9_qtrx2bG-L+KFR81{!H~waLHG`CBgt^qi`hTdxu#y}@CAs%YebBea}OFGN"
    "-f8p=vc}SoUvMYE7`@GqDa(L)f)P%my(l?EMTzfIkl+3n?Mn;T#%fFo^)Wc1hpb4R{8Nm%G6"
    "NypvN!R9XMO0V*XQ|Z@aq;ruyn`nW}kYLz<ExeY%aNz!-#%f)rw^YV+sFgddm|Awfni-e-<!"
    "C^DtPOch`B`TyrMqF%3;-L04?Cda|V!5>^4jHtDP5iXxTt5fBMq@i^4lPv|e&@pw1Q&iN6q-"
    "a@C7nG>C>ij`lug}B14$XSO-shj$m>{NHj&m34D5jCu;Mn{c0-0<Sl|=ZQhChQBA>bu?pq4e"
    "f49fiXaKNFF-Tp=Lm~b2SEJgy!&ydd)-}MwV{~M$;>%jh7%4Rk(C(WR+RSL>lVYN-^Ba3Tqs"
    "P<GOgk^?E^Hgja=Cek(0XI+2FrQIZ#!r&Iwh+EDxI8#0g}r?xh5=`%TsAsz-mqz->o@%BisV"
    "8r)myCM3fpt=QT6YOH`#{xrgFy&u60Bjy=m`$6?uEozSB6Uw09-{GLM6bX{wFzs?5^hVM#My"
    "XF{v8@ZIPQyRU{>Ro|)+SD}=yL3_0Koc4Ge1^yhmeaBjG)7&K2BHJ?VA67dCT^2TQRZ1?l6R"
    "K#*F8`Jhc@0=<*4eHKHJR8v)t9=Bj7}AA1f9~5tZE?rG&sH)@LmmK&ur6?8VwV1*L|IqcZTn"
    "dhejk_Bl|}-Gl?}XhdnihJvl{gdWf9t3<c>CsypKo;Mt#z{CauGUW!BUZwP;1-QnB@W2s(#d"
    "iNirL)v9J6~8(uK7Vyo$n-p3%or{ez>+n~lx2^>SIL`u3_kpR{2@&Y$zpZW4DoMeuyYr`>mj"
    "1?i4?f-7qD%1IuXJb9)87*l^oyBlVuqHR&WHmU(~hCm$9*HIlIucGP^J8T}_X$5de*(o>Y2N"
    "K=|bsb4F^;FTa?m!e^E-JQY)_7)v=q<?LT-fVn^<XV4T=4keR+$E%EF*UDWjTicox@+iKd+-"
    "@V%98fGPdV8P@2}p%7yT~<I^D<^~9>y}2DY4wjqdhgcrZeVolNysUzKU|}UdyP3HyM~8&09X"
    "KZh;@Nrxjb<Ra<>eT3U%%Q@Z(vtszBVk(0$<YcZvC`*iH#4$juIMFu*m(sb{5v@fMm{%FUb4"
    "Avm4MP(l<pSUR%9Mb$D^js$h;*_l17m|%R*i`mjYmMdfp0$g%jYmFXVN-BmatUn>eg<L|q=n"
    "lT6)eLX(#$fm|3{qWWae<sS0&$PlkjVjapnl|Y;t~8l4fSU3E2kq7h=o*-1_}BF|ByMu$&=Q"
    "l$*WhVEE6@)oto!JEQKDBA7Y7*PX<8LaIB?kvs*dn;R~~L}`t?vk~dS^_|RaZsD*bUH2^Dr-"
    "?Xn0uR-f(Xx{I#})=D(#m>TAn482^uqzKkmYCTiI6YY68rO;Y#;^BV7O1zeh#Cm*^YtFZfwU"
    "Q-FaN%@c5ZOXNPi4msKlw(v2%&&A>5L=CCtsrN<^(QcEyaRdJdxCD95CObA<g7V@Inr*3#!#"
    "QHP}y&PorXr5iD&+F=qWx<O`ty!5B;4oHp4wep&$EDHZ9*vHNUoKrBJQi(y6O(dtNmMq)&e?"
    "Lh%?D!V7thu6j^&wI+=3bLss!*B;##gJ^I3q`k|54oPCeP1;u>Xa<Oe-n<Y<gu;hs20_5a8_"
    "e|o+~?B9P0ywzpE%QW2z1ONBl3M&_LKl@^Ms%~3bTm=8W``=f%|23@n*lB@T6skt`IA^uZ7W"
    "b@eVQg=>V9=D|s%dxTdNS)1P-8RiMc<U0O(QN7ugt2IZ6m$R)v%kr-+itFgs`_3C^4>a`CX{"
    "AFXr<=PkanZsY5X<S|+%zxH&eAy^)73B9UA^&t{*l3D%jQ&J7DAxm|-)3OmK}fm^axM159Ny"
    "|NZdgLSPg&(wisy{HJoxjP?YVPAp}0;`R1Hx0#uEchNWIy1e)*e-Fu@0_qBE>^|t3Gub9WZ*"
    "A>kO<4=4sWRKJ||s}Q9y`8hwu8K2<v{%9#DAp)n%m=mczh*b0C!pACDs50Qd*+U-|$*KA_N{"
    "G27x^OgEE1!&$Hbu5+(8fCq?!G!rOW_&-+0BR)Ql&Bp=%DTD=%cTwnOxyCtI&F%=x?uGp_{6"
    "$n;yVk6>VZ#b;cT^-O_l8Uf0zx}QK=P2(xmAZZ19tIszja+zuMx2+C`>==o+rwBW*IM7hrTQ"
    "*3`OvvQWD2O#TQu+3&5Us5ArH|quNTkc#lIbgk~gr5lP`!S^-k_L`$OHLuIxI$q$R+K7Ed&7"
    "aUh_NF^7Vs(E{<3>jCmhhPY78qmo1(2P1Nc<=f5{%j7%9HOHFU6#Zffr_V$qn%0*A|;8jt~g"
    "y7tX@t4tSC{D=^@goNdtS!!oofQb82Nw1}+#NRt>j^q0E6*noQjbrL$!RIeow8+(2)SvZTcW"
    "NY*&8686`2&1tr-EmVx5Iu2>7)U%7jk9j1lm+}rj9Sw_SFJ0G9dT4OPlTrF>izivXMf3qvXM"
    "_a;*K8eGCPIt6pFaimX<$LXpOi^(Ewv<?A9V;|M))LI0M@1}oicSLXJ}7b=9(MY4EAqy^#BP"
    "?EmWh+jY<Y{*ay{D6l0h@QW8+LkYH2fd^rD<g888OzpCx0+wvlc@2Kqr^)kAabB{AiZL^hH+"
    "B6#j!}2w!f+MQ2Y|K?TGJHoz)Y>@h&mV|3KX?eKApWrHuMOA?9QmtxUDZ^rN+r%AhR4Tee}k"
    "0+Ab5wv^C7hxAFCSZp8|b46?Nc&WD%w$@gV65@rkDWY2m7Qf-cX{G$@g#MG%YI=q|b;3lgmp"
    "3&;i|lN&^4Nthc<-l~u#lqKONcV4sAaK)s^GKB+m&mXR^Q-`{5r@CQ;3xnwMMx|`xCkcsH)E"
    "v{B-T60MGD-IuW|oEaTS*;6q*)B|03F-K=n}{RwKAVuztg<PLpZr#c;gdvfjAm}7+qeO)?GM"
    "Jc*f_2h9^gR9q(j#I(`p@d7bj}D5nvjcPR8C)<YgkRN3JsKvD+DfH!7dATVCdrg+3#vWyDs*"
    "FN95Yuq+Kq@7Lecmani8Wl8+{=}-*K)?d2V~+M$9?Je^@FzN*aGs^Y1#4y#a}<dKMIX+ih_{"
    "i(%nGCL3p~G&P|Y~Q0K{k)TK~o#wq>VGoFy3L#i8SL=s2}=H1!E*`7-QAxQ180%nZc?7hr4>"
    "n!-g)1;xKjxMIUV)y6te<^$_uZgu+HoCAq}_FBQeY=?bG(Z*k7&Y3vbd7Y>4GS?U?5NR<(Ri"
    "U=)X#lYwo|(ObJ+3({fw5cT<;<<pQ${H&d1A7XC2-1$TZh`s>et=ob)5zTQVGax=T9u5H=96"
    "r16{HB6r90{Sdom!@fLt(t1zXzj37k|s^u%&a?@BfFIN-9RJ`~s;BtL|To?SZSZLg42~~F-q"
    "!O&idw%KcQjc2+g9MUSdJg79f3i`;9<^qv@K;qMYKSbEu0cQaFp;~*mXsM3Bj=neh!1Pl>5X"
    "EAhk)@h#?X%y(*yzp2SWXvH!HNM8ttfbOql<;JUit8*`uRqS-Hi1jl9}Vhaj0fSFTW%yqah%"
    "pX=m~B#*YV!KCVc8GLKO+uuAAk7lg%sH=#Wx4bcl1Zw0VQNwy{JzS1E#p^u9Cm%R(Pxm?Jp6"
    "0x$xlTB>B6P4obpL9q_)E*k26yRT%=LqE7p+ke?MZM;?tDes;O!hd@-I&{*51J*{V)iOkH?%"
    "CHYIb6G3C#bii$bAdO#2DR8;1d+)2}8`2cdZJNGIx#X<cFZCqzEV9D7LOq)K{d!pj6Y_QbPx"
    "yYqbnXvjJ0QK?%vw+QVN;alaCc|cP%2Q&Oo>G8TiQSbyc{33uTCmj29N06E%~<?!R_m_^PNN"
    "*%3pAz)y7(|1KEM*(XAvzH<i`IWgPdpM21AaCk#2z>h7*3F`r0Uwq4Bmv3C~!^u3|0WCDYOy"
    "?^5y+mWGeyLCgxUC3UHI1pxAgu>YAYfF4oCWQHyjnLJY{#p|ZHd7_=1_<7y}4w-2&cD-SXoM"
    "!~;&1pX`SZ^+#i`pp_r#EGCmOzUlb{vLaXZPaD&2%~p0|{~vX<F5^>no_H^=wpDrXQ;lf1#)"
    "j<cyV#L74T02HHri-89-@&9nabN!WLICm<H4oo6FhsW;xj*Nc%FtSANQTpv>I5{h@8r_t(tv"
    "|fb8o}rnN&lzee^XHDwj7P=ARDF=8oN-N~jFK}!XF+$=%@}0p&^%deq78VBu#Fq`yXRq+rrZ"
    "R)766yBAtnbKXD7X8)ht~syUW7qRR_ws2akq%7?oSaHK1!tv<U5+&zZ?grnH#^+KA4Od3x)k"
    "*smkdEW}JiJsMzcItx7E$!26up*P>k(<kL-mTfNwWhPED<}MV&E%+@)?Y`5@!|uLwcy@AfcI"
    "sV?hKC<fb9y?y8V`@h|0!+bt}fuzZ=VHN5KBKvUUBe7rs}Q8?W;-#*Cz3~$X}=%qDbM2iZxh"
    "+%-5X$5(xraBM_(i%wg4Yl9MYjkhvp5`s6DE^x7;az%~lY(Ol<j_eNiN+dS1>=|#=gR2ynGr"
    "$@%zI+4OQD6CUlxyB;ZWL(Pm903otFDuN8VPZ&Nvm};8akYpYOCoP^Uge*LGf6fE-V8T;1Is"
    "4TN}VNjf>rLh)m8Sh){!&=EjYa718Ao0eGT9w4|y#pO{Vm^;A-n43$m&K(FJ~$_zT9op@G+x"
    "-z*1-V*01vAt>_i=4O4X?&wy1+iXl+3b`(pi13*yUM$vkB$EfuTuVB6*qY%-Q`@U_&hpDJ(W"
    "kh2K^YW!NtSG%yQ^w4T}gq78!hHZOcd&jFm!qL>EdwY{Wkh?X`Pn53a6h=M#t7U%d7G3%hl+"
    "za6-2LPMyeXwU5J#(UGm1l@PY5uM$9}+WSwJC{Gr3yM6tiM?fY+HpwTCq+bFCV=we%vMqbD9"
    "%nXin`KNe<5*^~{zNX6gEX-RFNqhTM8@Lo&uFj>qj4H`nk+Cc6rjgPs~lVDXA?%jxv(swp-t"
    "p68i{po=UIIGI=>jbAOAkOtkz1gtK6lT$F5j~ZP*p7@f>!Q3!HW4Rb$Aq^KaJqmr+56TNpI("
    "I^3cVDt2o|Ip~(mBS+2gQ)LzI{5~PmC?+{C{pl*3r*^#vZqds6H5PlJey6G+lv1=7rM&pwD6"
    "OfGfPt`|1Sy?@mx5}}<4=N>CAWDYtS>rNmWNd%*v;p+448QY|8v2mOX&H~nk(?Nf!(;5nt(k"
    "j>z^_EpSbqnKy00SILKPC{JPf<6uMmjo#5+Y=vj@H#;3F^h4dw34FG)uPwLtl{F7FtWRU1TN"
    "F43bjV%ves++s@?+M#MC;$hv$Ad;e*o{1_k5107zJNS&dU<x?oea;<$EP1M9=WdzJVvvissW"
    "MS#+K}oDjj!)S&_Ie&P&#(Giuvx9BgqX<fdBBjgue^Bv{{=Jcn`5oqRA2zE^S*UJ5fe2o~7i"
    "fMDVKEE8dDA^h;$n+?JaOKnIaR9J6+aN%*6C;R*KL|s@cUB>oe`hB&B$w<h?Qe|7ioBf}1sM"
    "2o8^h<p9>Z^+2Ytzspjl8UvdsU6O@J^QPm$_%ayY$74J*iZRc9kMGsn1zm)tfAzzFs;98+Tt"
    "5n8$LYy~W8dkL>r=4XxZ7*tc87lwBJ8mLMwm`nEJ1g=s=UU3nXYVNuF0yN%;{#nqVV<t4T05"
    "oYrzbSDistW25mwDNo`8W~G6>q@&(NFo0jd?C}fX$WHRL4C9VC#*Nk1~Zl;+|W!n#K>5j!5;"
    "3dmoy?>Y2!21A4M}f6Tzufo3z(8MoSHS)O=OtL06%4-fl2m7=Ah$UwQAx$D`BX$w=|Sw-4zc"
    "=eb$Y80kU5GG#N#=SO;4WV}9%0-Eo~biPh*vsLKhLZ2+7QJKR)7NAXRa2pZcQ`Aj^3uQDsQN"
    "nJ_tfm|qJO&uKEX6u^!sR5L0_o_+$QP5URZ^l-IPG^IWxfSp97@214I>s=x+4xb4;#1HfJK?"
    "E3hf*OX5PCc^=S%oW=S*8u{(b;n_|o&ZgtCk6`9>)hU|U(OT&KEPyIWREd79FOzYD$!P4K-@"
    "P42BD;iIGpT@3U|9I=)g?k+}th)e+M|Ojfn;aQcw|6xtoL-g1*3zmxN%sieXBvl*0q^Rb?dg"
    "jm(^Jh0Nwlo`a0&~lOe98m2@fmDr|BaV)bWHH7+yR9_CCXXTuV62x4)y|6%6{y8z0p&m3d$n"
    "j|bMbRUcj)e#D15V>25ND*U1r3`M@#gpBA{)P|Wu!u9W#P=$wC*)p1ez^gT?fkpz`L^UXO&9"
    "D>0YehD&0B!TxNTV`YJ(G0mm+7&wJdNUNJdr_MWopw3SCvwXA+xA!RW>FKPq9Q{#NrN<g)F8"
    "NY{L*nhE`}eY2|^lF$tFek1+En0Sxo%0!e%BKa{{V5B19{P-L5r#Q69FR!^)U%!R|lvclny>"
    ";hB%1{@}FOY<iIE6r~t4x)NnRegowimFiCKyhruz$7bw&Y6O;tc3T43Ol3HjNepgznvU+3mt"
    "8Bxxk8Cc-XGVgN<ySEMAh%lKif=!Q=WM-{wzPMgEu<w#UcGz0;N9VloT@172VT=_;KY{DxD5"
    "VvE*0ZvlXItJS)6Uan#cC9TbC^90Sq(P*u)hgDlq5tt@hp1m<EsbQ8d>V<*Dsu`A5aUwN}@~"
    "9Ho5UpvTuH0ZvtO+B+3#>beRQ8g_U($QLei?A69|d8#qSRugQ^(li{>zOu8~@2NH%jyl6IJ;"
    "!gwmatFQf0bM7tc!TSRwv!M#Td({-f&%rT**Z^hPxOD@lV)-280k;+um;<V&fUh0~1nWEOjp"
    "0c$s6Yq!Dzuv-!$u2t>5j{^%4nK43$0Ug2P%t*~1C1Fl4+!;tQzs#Pl^d}oEn-A5N?(`}f65"
    "Mk@?;UOh3e5Uw)XeP$$*7RfDMX?apHARoFZ1<=3rh3qikGsnCEfM<HO>?`FtIdbJ$jXP`hl("
    "cxheEcezghkc_t7=~AL_`!ej@6~c1sbr*+EOCpy~%o6lXvYt+7Kd2~BPa@Y^hSo;DkHuj9$N"
    "nO@pNSkwR6(7%835=?P)Am@uy`F-JZE*~wOhEvHZ3#JqkWwjYb{5J5Nj%jCaBvJeqjMO`008"
    "T6GG#BBo!m6C5@8x6%DNsMIzoKOF~(keKM7<9Dz7R#FEdZd}8SaK^wf8N0W;Ci(55gGT#;ie"
    "6>Xa<x-Il5BYLUG~Yorq=eCv={IK#>emCOsmGjEM)&Ihr)bTcEmf7x`Jxwn8NpzmF(SMo^ye"
    "aBrX^BF3VKZiRVr9ev;H8C7tngNUa7Hgm29WAQc4$vd}t+Bl``V5>SJ4N__tsB81mITZ(Qdc"
    "LHI9`M75%f4AVXtM}~aT=98y3XPLxJM=~X@i|w#Nmm>~bS)6K*TF@eDqaa>?RZlj64Eb0+;#"
    "`;aDA1sjQv!^JEI|BvC>6*{!uBSG?TrH48wIvE*VR-#Ku2eW^(rT=eN{!XYn!A-zo!a!wgil"
    "nGdhjX8Z;>w?v-3FVh(0iiw>PxbKogQ7u9TKxt?8VSMCF<zMR2~KqhI-l!_h9YF#o@NiJ%AA"
    "eyf!4$5B{>x9fYbAbF|Ae10f=}4v>Nu|8ZT#9QwPmzzxatb{pWGZT<RG`RY*Z)n#E<3Fm^j7"
    "~zG+WS+b-(0^X~G*4PQ_F(GLAnMIy9u3yb202W~7puC8MZi21S5K8ZfHHkyF)!26>u<&xu(@"
    "0i$0<%6rR1ytlvn^7a=L5>vvqwfQI6C`G&wHfoNCBw_)23;vzXJKHTy-P#PevLHfzD^}TfO<"
    "_Vc8$>cfbxcs<A`a2OeD9SQUbUQn-`Q{%ROC=2zHf}$P&VGX5KrzYD4Go7X#v4?JrD-QCO=5"
    "9Uf~hlIKS`SnPxfIy7_E(7aCcSyIp-Ftr>y#;)-z%cC7~=hyPSg%VY?aXXBP8#GEbOa#fTn2"
    "e3!^p%$_cAyQEeN8_Bcj<lc;tWiUTV-E#Iz)X9$QL@7A$6H0-Jc^c>OxO$$>bmY|kpKt5Yf@"
    "1G3UT=cBZwrkkT~Kx6R(g-oH&(Xv(b3_8U9q$JoG&9c@hQRFogm>!Y1kJ#Yc6mk2=auOFb?n"
    "zTLNGF;Ihbq^<d37-PWn3RW~jEGpG`q>i)8ksg)<UGOEGiP-HaBX!@BRq&QRm-kngW{QR^ON"
    "$ppyTef_CFR0mNt0YIDkDOn$*J3H_391yU(<QST2F3zeMies9RUNuiX)n#K<STVC-B{xlZDJ"
    "rsF&tkx$<GZc@B@M`}f5o>-XvJ3s=oSKv@hAB})?{$sBr~ozG~N5=hjU^8&v*Qy&|F2h)TCx"
    "F`--$bZqN)TTfiJaL&vMGMss2bVaMWk+!qQ~doI-o^cGH2(1M>e4$qJ^tbypAC<sE}7A%7XC"
    "6d8_xEIvDw`w)_$OWSY(7191Ywp!nGH}Mw7<PXBYyLC25J*XgmP|)QF}4H*N0}Z(h7lx7ghA"
    "aB(yl7`>vTeT_~(k1x(nY1Sq*INDd4Ax*kQL(HWMPCRzczg&GhJN<ZeGIGuL>m}X`<lY59k5"
    "`|D$Jn<^tz;^aSWMOJ`S9u^n!34<vIsEYH^7A7JncUUx}bCY2Ft#wId9%npUl01!Fqc3>2h>"
    "&Ilda<wSCiGi1$e2rKZ$s!Ub468V|jz_wAN<_IY%1F+QS=Hq1uVNj^QZ(4Ym9KT9qwnXlx>c"
    "W~;WmdfjeuzPX@s|9WyG#g>Hu7hwgyu2D+c*CQk3)-9DKAB!GfokStcY!%cj*iZuix9X`h)g"
    "(kc6S$Kg-|fw?fCc6EOfpDZ@6(z$A>U0<99~Kr-z5f-uRt&G`<|ZJGLX8mbpAT{0(pV-{WOH"
    "HK~ZK<<48@g^hhUK0Djc*pt4x{l!BiP5!)l-}<pj?|7(WNN???^8OClGSC|R0o;9ql??8XD9"
    "bpQ&;VqlXdc|+Q3z)C8^OZ~h)q)(VAnA&;n+40U8QOnAbpf-JR#Lom=Lg)*vAuT9Tdx9C?C<"
    "bO^V+QOl%7&;2~kwDt~d7F`h2u1DchCo?YH7*O+d8@}^pKH%neh!WPvkX@{jDS<o}@SNtM4H"
    "=QJrvyhDwUWFi!vwK2XsY17Dwc52BtDcrA5uqOQNK2IlmD~)~2ult`BTJ=lq@j&f(S!DhQUC"
    "MROqx8z<&ZhE0$4?r&%i=G%x7+GKnq@$o?&vMmKo#J$3+sD5*V^wz&8Vt30aj{Z(V#ky&9jO"
    "Ee^GixjZ@hZG^gG6eu|ru@4}T%meg8^3hsIyulz`J1Oz)>}Yv-C?Kfh0kJ3%lIr3v0XB}VEF"
    "mkP9447Tds3f}dv6KLRDKba#qZpj_t6g`OMk{6-0p2>CzCsc>YtD+wd&2f*e&28p}RWWLh8*"
    "%y_dcT2@N9EuD8_k@U$ILCRlIN=TAGR2DIcCqFO<kqlI&<O+<apdCQ~H8d5KsIGA~f=sisfA"
    "#VH{tedUAOK(InS&3-5W~<(J%fY~MBXC;7tYK!yPj4Tyy=YKcvI`*;odN)q+&CZ3KPAErIj="
    "eF&hzIeHino4i5j|9nys4CsMefHpMUo7^Jzzv__{fX5@TqyG+$B&copBRpq$7wq4$kd@Fl*$"
    "2lA#9BQmomDtjvuPhDu3Vt}c!s8_XvS-5zIG=e(@k5V#+q~hk(r3*y_C)c2?KK+FumHy%<A3"
    "AAf8jbwqPPy<KIW^|Mx{@@J2I2>$B&uMcDx2Iu$!G_WVLyV>DlGUFV3I#5zg^J83QwXQhY6U"
    "-dd^H4sYz1GChDjG5=%ah@36mkStI>W-7?&bw_R0Jnc;BRH;+M8QGkP^;RL!u5@vJ5Vhk{Nz"
    "kUS{0CbLjTKK2Qe_q!dc-6%3P4gAs|JedKpk^8f(GG|Yakd|5W`Onj(4<xc79n0thTLRH;q?"
    "^?Fi1nRL9E=Rw1y-`^Zl?&1r&wTpth)`#NgHO&^#qRiYqD5FazPDf*HIFXk|v>#Jh>P*>!a%"
    "h@uz{ZT?DW0{t(O3bxS!HbP@R9I!VaWwl!wEa0rtokmP?ML>no+GYDN0rLN%QaxGU!2uW*N8"
    "kN8tWYe8+I56eCgKy`W(<&Iik9A_qLr=I(I%(>tM5ji9lFDc4&j&|GBsKZ1Q!s8HS1t&D40_"
    "m-?N>SDpAy3mPf;<%K&d!4}em-bh7CmPGv06Rxpp&p+}kOfkLt=-HJcg;)3%Bx}kcsc5={7p"
    "VI!4gJuIMmc7nh8G`)jpQcz8xKGGGE%vm@N{BygWtAxNKZPScQh<H=Fvw~gBihV!B;fXi&`O"
    "L}*Z{P}!J3gy5%Q7RqV`s8ykPS0m8naYlfPgttXzDD`BpZ7<Y5uq!buZA1=3`v44RB0T;DB|"
    "%A-r}nSYxbuYO>rA&(o%3N2Q07k$IiQ<At(HF_IBJ&lBzOck3uCuc{aW2<ngN}QNg7FE1V8b"
    "kpXxTX;P&Z)d>lSi3+S7!$!QT&kpGK$3H2DZ8QK@`H307Aml2ef#=$(F>Z0E~W}x?1pB$C$3"
    "?^DKD>D~v3{n)8{LBC&vv3z2+>Kn1`e`lr9|yzexcq6=2S1c_qm651SJenN+o(Z%IXMmiRSS"
    "!EKV9Kg2~klAfdE64{aehxTl>ewh%sB@2q<gGlcdvI8P8~U@@qe1C}TTz`RDB=guI}+az4;%"
    "5QB5_C8M!EMADRqWg4~a-h<27qZ>RJoMQjF#lgs<L&;XkQ23VL{gRhymWN_vV$+gs%gwQ}M|"
    "fGS<Iyr?L38%5s|R#Br4*NFLUTc5YY4`DJA-X_f7U4&XD=F9{bp3#%9Ms(O8U|pt@W+{<H{^"
    "cWaDO_QZoem6*ONz2xrm;?ZSi9(DC*tDc*c8Ny_juvJzp_0*7QrndCC)k(2rHR}bVqCDxiH*"
    "Hr?^<zBXOocAhtAqGRJE&xnvd6sdBA4BNKp|3b`s_YGfp}Ppa-hyk+vev`G_>QNticWblRUh"
    "b5#JY4M0cfzhf+OqLG94~5rC+Ui@%m0d0n0y9`1aDP_r-=ba9h>5P^B^;szWpbCA>~L=w<kG"
    "HEA7r-gx{t2fZqa>?rZrNH1+)g_UEhvozswC0wCgSvAebE`k&^NTQ*c$;qBT&&=C$4^o-Jb%"
    "Bmwe7+zzt|RI%kix7oP@87U!^l!xN>P!He)5c?iW`5xD)0~n>s3Cw?;GOu8I!*fdS&^qvgZ4"
    "~<QqwlJ8W5><hkD=uip<ZGrg~|tY8iJ!Dy#m-%YSlp@N^4da4^)~;T`g>glDP~hlrk&0s0|p"
    "oC%3bJnkeOLM>~p_o^?+WEda-ZJ#n_cZD~g~Y@$Z*&p?rE>KUlwb+gE_G+K-_0l63u-jC)Za"
    "5<YJS@Vq@1!J_~5#y|*<47!tiHVismlOtME0aYt<U?kQDmB+(g=<{yUM%YG(Cp$gitm~sLkB"
    "BC4UaXLc9iB@+c)3Z+&q$`GJwDGI5ujpe=0=zIW75~9j#8uDvh`E?z9<F=L?w*Ls(k9J8gvu"
    "0rM~gr~(qWXL0tZI0PtG!aRtAY<U^QP_CP%<sL;E8}n*W?O1R?_>RrRs{{pvimT7QOAirdnF"
    "$=-J!FIMp6NE}sZ#?3_c`qW)ybs&Zzg8t)>PWjV(b#pET?ddv~8fkUXZLOh&&aa)vW-aBS1w"
    "Pw(d($Y8HTrwXRGodo0JRO_!WhQ7Aj3JIa)CJ)Jh7R1v0iNb?TWX)sDew$_jGZkrUp7EG|?v"
    "%|UxXVWQiCw9ExNI~dnIwDKD5slMiK&w6>;Ji>C{8i1;bi+ZEmQ+rodAHt~GpgvF1=LZFs5M"
    "JuCCpooG9BL<JgSoOJku^Jl`Ho%Lha^CLd+DERRJ=uwh)P^^kt1ms;r)UWhrG$y4Zwiqva3t"
    "LC@@BP&4~5r<O+X+<8x51=dh9y81v|P6{mov(t<2mT2Nk?j|G*(Ug+TZW6?^CAAFFY_oXjit"
    "U;VPI`ak1hdPO8oRWu|HD#nkbui?q76lfp;o0>&DKnfeHerl#pX@3+o6DGQ<h0nQA*s(dyAn"
    "Dzvi{t4H*bAGd9`ot*LHJy{=|g)Erd92Q>H|9w;Cg_O9mLv;F>AjRiz+RDmZi<0t?zy?*G=="
    "OjF5zw$0wF_aHJ;a9@CWijI=>U+G_4-Z_|tXIbs$u>ze*y2WJe_!YhEV2fCvGtVk!c{BOvTw"
    "7Xz?WQ5vJIfre1I2<wjQebe)01z`l+fC%WSx_dDIE}N$91i+MnI<%j^6H8Usbkj?3=#@i>>`"
    "mB?%HM7YcVmp<WsjdZKf<2JArR{E-mE@zG+dQDH8*Bri-^v>x1FY-=ja&gNfg#XUe+j<W6t<"
    "!Gnx!29Xq-oQ)%CpUl9H;@?#+=%@Y0xtX&}+b5wPcV6ca)G9&7g_Xtul9~iF0!^c59R*21Ol"
    "8JZga-d+T^!S;w>DQDqWbA~K;Cbr_XPEKcaX7PQFl%dnmGvt}jhG_@BI7s-$A?P|5eO%Fd4_"
    "i0CSC#5xn1sx5cDzbTFSgcF$a&+-|bm4tGyS%bgq#EwrO6O-6S0y4U?c`*^P0xFe`9MAGXC&"
    "Z!U_f4el&RM3(bTL-PFct&DY2rwAcBQ#2$-e>cmo2Rg_)>m@G1=jMBh69#bW>z<s|<(8R99~"
    "siTwT2x->!Qn9i4IYmt)pAdh}?5AYwvQ<v$89pa_{i;KC&6r<dbh8ZCD*>+t91mtTfiqb2eS"
    "zBt38l2j%NXo4i)2wUKT~>nfH*TUU8*uRRy=Mdr8jp!LxRd>jkq6#?HzfSaynD5bZ(YwJRZn"
    "p_2K-JcX$TEx)$y0hewuCa4Sk)jHyla$?$h?bbfaDQH>y_rap`2v)}<cp3WBjoVDSaA)%-Zc"
    "8ToX>?Vj}i%Jquk6EhZ1%1lrL)?<5U^=$T(QQP0%7AH!_hcIp3L7+@ndI(3@6`OBfw;i}zi>"
    "c8VNUg%sS3sx(Zj;K1bJ${`lzGR;aB3G=>lH*Vw)g*)^V_w%&=^n!BYj0D4W^B*NfRd*TiEo"
    "E<22G;P6a+fxD-|+x1-5Wj6xWl)eq{D+M0F4^7F;-H1Vk!8D<)5>ai#{vsQu+YB4n)&f@I&+"
    "maALjxx5)+rZgjY?89VhHiW<IymU{KyXVXwR-y5paTN$69ywn<uO2M{CkMDLd$|B^nACHuX|"
    "o#z<o>!!~dCSmm^Jkk9gS2;;!=tY&&B7ac=*(2<vJ$SQ*Q{!8$3Cq_jsbtYAVn{dmGjhiqlS"
    "8@whK~T+eL{Jf?66b#QBB+AGE)t<|<^~AbaXAZ{R<-6_0yPkLPDL@LSSow#<ttP)q>jwh5`0"
    "q~9GA;jgM3&C?_S6`uvln}GpdP@4YRWR`Ug$F7X}7kn1V*=6B>MwrqiY5flD@B@3dLX7a{Go"
    "ghN8{=>h0EN{Dt@|Cn|SOVv4}ke}eQzht!OQ}kDL9u1%}X9scLo0j?7xK?%tIJv1I-ZSS!^`"
    "x(14PG)Yy=)|J7-~8YZ5Ym{i(@T#HxF6QcmY3TnmyrL)|GnpM5j!xvK~;I?PS}`P9Dhgq);Y"
    "^hS|FkL{kOSa=;V2YYqZHAh8V}ccP1vWAAEs`I|RBvP3Kr?N+mj8~X~(!+{YIY<#pY1~m({O"
    "(i2RsIr%`n;P>a#f#UNu<_A$)1?eVn=k8BmDxDCK3mCoP3>gXi+`@86{$#NiClqInzp#g5-n"
    "?1Ek;WyFo<oKo=3cHqrP0-y0378V1Y){yFS@zip)m?o}7~b7ia14a->)>h%Ka9tC(sNMV~9Q"
    "zP7OHy6y$HFvD_`(brI4eGkZkooshPS-76KS0QA@tACmqeRf4>N-$=k&CddjsxqXIePUzyDI"
    "2yH*Y;d2TTg|rN>vj$KsKn_8!53nZ<N1tGFPw}$-CXq2zT?5D=X}#%;+EwLQBevJ3P)xqaB9"
    "fooC3-8InK9JgIFy^&Di@Lv80-l6l_QHCe4gw8eVqsxMmSmWB8_(|R<%yc!=LkB+>{t4~K%g"
    "H$xNe60V{*_C(~YC--!<?5AG-YBm!fI|`%&mvw)r=hEc)woa@H8Ix{XU=S5#UtOCqs1hgGM5"
    "jpad69%6bkgH%5~F9&2a>C8`Hd8EI#&uvQfzXF?9pIj5Y44S-=X>RmOpu56;jCZZ{&;35w!S"
    "+0iX-&iZxD#!y3AxWh+E7Eh|kAZ^}gJZyUyN8UX*^eqnImlx}C3uq`LF}YdK{n!?MQ4CuT1!"
    "+VCFV`RJ$kb8B=s^a)VygvDTRep&75Xw8JTGR-&1N&I_-qe1Uno3h!51;;x00P%&q8iCV9Yt"
    "^up_7Kpd-(e>Pbg&Sg+Zy5lS5z)f)lMG;^vN$1OFrEK6#UE<WWQ!oHu5Fq@S^2+i*PR>uLQA"
    ";uJyb|~c<g^+w@@(c@UXA-5l`zCEEMU077?`$$<c_185_)>VwS@6w>>iv}2p@aZZevc~Hesp"
    "2H%q}EkgAF^Tgy@+PA6;$-@+u@hzN8hs*?nC|QZK{WzZy^FL@16cKE!9^is6Z^Jt2gl?s`GA"
    "_(4iE<r8Wke)S;CWLaV1>!Mad=N1ecV>%e%SuVbtd(yEWjUS>$!4jLDviW0#VCW?E_Fk`M^E"
    "y6dPI5FZD>8u8ua+)4k6m7S8XTSmhk`+6M26%+`&T@P@i(=ZLT`mq$|EjJQJX7%qb_A%)-Q%"
    "$Ndb6DRpo2ycVUd3JGj%)BQzEKlOl@)M`IJ}KJlg!0v^fr6E0I)z{FUvv*uY=@^yn(%=;J=s"
    "HV8QspV{m=h8c8UX{`p4VjvTQR5JP-6;(5qsLfe85QawI(^P2YrJX(l&S@GF){EW^yiW0COL"
    "%{3p9%KmVkc@Bz5qJLwnNeGIZ%30-4t}^1?2?5bbyG0F%jT7Gx^Cl`Ej84UeHu8{RG=a2F70"
    "6cu>Y3(tK_qOCi20?ug85UpS^z1fYHD;if;MFRy4J)-LABpp1a-=A_ygQ!e1p2qOy;PK+>cW"
    "NpAd3fxNPp?K7@WoaWPsW^3Cn!`gH+Wh7>S8!P_1+Dy4nKOA<Nq9a=dgc#IRB(Kg<WrGXoP@"
    "lYG*}rp1rssw1)`c5WP--@1&;Ol%pxdeo<^KU|&4^?fh(vU*CW5&dxDUo?>_Li|GF2Xn1ryy"
    "1dL5yB2F0L(9#ncro`EB^Pmy(X(huIonF^0AnOSiq`q_m$VVVE^jm8w1+pX5gIYek<*PuBX;"
    "pdLBDvUl@p`erK2+zoIdmytkHyMnU~#Gqt$J+ntgqNhVjF0%Db*ri}!hSb3^!?@(JFT_m5vQ"
    "0WE{?VHs0)#dzG*^D>J^#cEt!opZx+Wr__WZ7tXVht*}#u`B#18o&dH{Imhzs@X^f$}+_yj2"
    "Yd4RAH?bqS+d?ornKw(oSC5#`8}p3&DYo#5C)pA`5Y?9UBY(VHF)if)P)t<e}nOZ0f1llu4P"
    "Y^s8!mSWQ5S2AB|9I4FgRAGP5P_Bd>%acITWvIyfS4RVzB^mC{BroxNYRFp_0FJl7S5EHk<7"
    "CXbZtl1c?`FSoCd}mdi1^lZp;;T^126I12^tsKQJ^N_@+ssqkX}!Fb{iB$nBR)g4aWQ>2IAs"
    "{gSg7m7JW7R~qNb+#`An)uohc1|pW1sX`X^C{r=lpaew142T2`p!K-a)16RJJO<N%zfCU1=e"
    "p+LbxMi@0Y*_iDKcok)gc_2983sPY?*ryo2hK;vqvJ@@R?gd?*xW^W3DAU*nB3TE>cw8+9i8"
    "w)-?TGgvEwX?VbD237Hb5n60PKqkq8@f_jj_P<tZ28f+t};vP5iwky&l!TVaf9r7{Gx>8o+)"
    "k8mv`&)>f+9hgGd6f8;Gm3sjjn_V5hELXdY@I-#Pt^(HTp{;n9;gN1Pw_%S2a1BhZMT&I{pl"
    "@-d1(dFsrm0-Id$thXwb{o2?S&0-av2BT5Em&ABweKw*t@x*e1qRfNkH<K3pIg7vRJSyYC+|"
    "Tni)vvtK?S^)PF#gGFVBz1Vq#b}Jw)Bv`}gC+F-Yd>>%w-}40juf=*@x2QiweU@1LDBo=fJI"
    "%$MlsD<XTc!#$5b>WD}Jv&F6upUgeWLk-<QW{P%nD+gJU8xW&U+ER)VA{PO>6Idi9UR5wSiC"
    "$N?r&U-)O;Vvq@1RnlUXsQ$&t$e1#}1HNQ-4QszF3lA-x<k`{}O%|oEoxum-5tZ^o%1EP)hQ"
    "6$<^s8h_T9oShXS`wY4evlE=8v)K;0Mus>!o##txNW?qq?7TNw&2OwVr3k8a7*0j{!3s1ywG"
    "89_bQ;^(<)nkP`2<<Z^k)a$H!{$pt{ItjDH6f$9Tv)|)tg6d`j>mMlrKmTGuc;RQd8wwYE(V"
    "ksUN^n8z^KE|Ym5M1E4{P<aIQfS$DzDJll4rj%gs3_*Qe3c*10xq^)<f&sI2V8%GA?+Sg*Y~"
    "L1za0GPFDofMed;xt8G6a>dA?IWkWX;_<6%6rc%<av1>kv9A0^9w?ACk<+7^8@@V^!YtIqPi"
    "rC?Um|P{_ko760rgJN3~eokVIdP^h?~g7XliRRvDMEb69Y6#C`SEf5Q>h-95NAitAd@?^qrz"
    "c--xXcHZ8nTQI5Cf?T>PP)eoDhsg3P0n!kUhUF4)C(R>flCrMV=`c9wvdF9;br%#_;-Lg76H"
    "r<%$K({EO7hzMh3}o9I2y4!x+^e1(69`p<i>g-KvkpW&W##3{zex_h8X}?6iJ5K*d;B(AgqF"
    "wprb$~mIyyH6<<5umPZ{Bvsd3H6$sGoIHXgn|Z@0232N-VA&sYS<VTS;b=Ox9hNc<x1!)Z1&"
    "t5m!4W3(xf1*fipaBfHYX!Q(yqNm5SCD?OgcAVefNa(Lukq6L|6}$L1?Sy=cR|6s_>O<6beE"
    "3Udi{vppl(5c+f)*RJ3_%x};&Wc`_<D8{&!ci~NulCEyL*t<56Vg}vVq8D2sTRy!ILv|^pu)"
    "3QU=LGo8r~X1W$GfE7V2)jKH5J>ID`H^n}uMjPYk#n3&k5?HS@Ph9x8Sp1<_sPQz>bz20kZ!"
    "ma#0{I&XbN~fl0j3r4Uc5dWkl;u&nloBz`?5D`pn_)^p`X$x5D0>!5yLTW)t!$POf;x+4t_Z"
    "X|S6^xiwQt0N5o49<4CNy_*a~f+eiKLQWu;lI>d`r$@6F?;k@nEa4AzdCY8>y-nOQf`hl99N"
    "s4&-+=c_d5F*6~ia9&$n64IvOynqv6b@$P?#?n@P2JpEie%bv|)7B|nZ!-GRp=73i$YNfwv7"
    "3WXa{EHV`CB}PsEIu=%>MPj?n1s*_pB*hw#7$AI>F4m<_lsb3;NedWAsFTz^R4~KSesTOk)1"
    "bale~_9VmBwx-XXQ^xf#+D7DEk?&26t0W6&PH;c$H-(dT9wA!UR^e&8emdWv1QW*U2BH$T2H"
    "JA6;ip!Pn=<C;#=U!j5vKpV3kP|bqq9!lqu8ADM(YQ?dw+Uy=RF%f(*msPXbKzVJEC^E*E#q"
    "hv1regkJsOTLcB$Rw?m2vI_WFa}do^vHQtyu7R+UjozO_g|Y6EobcyYv{m!zvri)||Q>A!H1"
    "I3{=Weg$4ci&@~$T|@p<xPI0V|GC%8xeQ-bjH@hPTJ7@g_pRx_rBetqoFE~=>wze?IP-9|jF"
    "K{0H^(V51JNvkHbAmre=^6FXZxtCUeX!CJBBm@kBrCe8$N!R<~5p5g{m%N0G;ZuOmKLhm15m"
    "hZi8EEv5H%g`m935ly&H)Lvhv(!8J6$n%YkBp-|{X;zOf{><j`G&iz=n$uFid4Oxn&-P_^X5"
    "uxTcb*4dR)4y(pU~@B`ObMT}2^C((;8uiuOW`-v6xVQLQ`|7JL7dXMIrlM2<0#dLQo4lW%KH"
    "@K{#W2eRAc38_Cn&|j-?Hxno^U7z5SdiFe)jno77|~!?gV`Etk6i+Afz;=<F}7;c<m0iW2CE"
    "ZWc*B^Fnj$$XL#ZYBr-9RaBvwerOzhFC`O};+O|5uocG~*y;FaN49tBtsm_|-m>d;tuKu$Ly"
    "u)mkcGn84GPQv?C^5rjZcOjM&9t`Xs?sncN~pBz(*Fj+%Ygqtz0lt@y*M67SUa!5?dYm&a9*"
    "j+^A4Q7?!;my@#?-v$5AY>Z5ryK`~p}6$?<Xtn-Vr$?VyrI;ACQwhqlei(9lp?lo{MjpW!nt"
    "6o>a;(Xs~X^t)h-0p#x_}D7+9JQL=cZ9L53IXwB=Dh-fA{1%V8;yp_tLf<k!+jHP9_u&C?ET"
    "%+@)V{wm~qZ;9@a|6-&2j^6{M<G@oI-MY6Z8dV11i#0V4}%RPe^>TW?bgtrA$Yb5mV+OPpqF"
    "SJpwrVm$jH#cep|C$o^y#DVv-a1D~U=D3|$t>>ga*FKBgW{PXoJdsj*cG9@asf^4Ac6S%atX"
    "Upra8Zo@GJHkGDLgzoIXO$+(8j0Zt1+fo8ln}wcYby}KKzp1>R8yyO|{XQ?P7R%m1Q`%F|y="
    "Ql>DjT^9D#RM(@YJk1oxPF?Hu+jEl#9HyXP<`-B<We;a+d+)$sUUde6X^wY`cctZ<<|Jqx4_"
    "vLDo4+FzY&KN!pFGfejV_^03#=sgV9s_G3Zw#!3!Z8R6oUUnRU;9|zf4am;wTvbA^nkKr-aD"
    "fWe+}!^6TwzJzK3ogu;Hhp@s&5cxEj9)lC4^sR8$pZ8pC+Kx|gDkss%|Dj3J0p+(iZcDToSy"
    "ZP7)#6<pdA8MvmknNaO$A~KCHXmS0Jkt~%c3aR;`IL=pXGM?eVU_Li5!-UDZm`;AYD`^|2#v"
    "aLp7`CgfX;F*Hni|3D*6P~^1Ck{(H2E)#AC%Ed@KXC)&N3tbnX5JvehMvTm(j9fScD4VQw&1"
    "Q(Cj%i`2xX(@xt2NuGOr`kET?iU}|D=;~g<o!!SWhP+(7b2C?0c;B_GCWEAxb_a4w9UX720J"
    "qHS$^wEJrW;BYC3<Qtv1O{Nb5=UGcl#h3%bzKl1#_0A{V*k8u%%2P%vJvgH*Ex~xXzVE3^1_"
    "BHF-Wr~;||8vQjaRkk>55@G=?OZ-7HL9zFLC%q9<96es=$=Wd*&15@8N9NGsv)p0jC6yZNu1"
    "SN=7hm@HVit>kcl7q-MXLZ%gt!>_JYiDecu_an@V3R9>citkLxmU{BdyN_YtA@(mv`;lkfok"
    "bepfvHw9Z+HM?Gwy?^kbR@YgWtDoE#)n<X=sl$B6`J?f{@w~Id3Tmzoqw2yX^kyEzGd9MAH&"
    "XL7Uof8&(kR$;pUD4RPk=#t$AG7CeN86UcL3V$Sd$3#+t<#0{cH-MX-%8G3|oEgFKA{UvK-q"
    "88e)E!Y~Z+L!p&`-2I}h8W8>P;#nBcB<9M^3<HY11sf*liM8Hd9xC+s$I6{ZwuPiP+}~_dql"
    "Z?V)I^MjNmo*jDmf`ws++4l+2;4su>Y74lM9<ZMl%L6junu<DpV%)na1N7pGumg^N=yRAzA+"
    "jaD-a<t@(qu!S~I$}CO;lv0aRgYuOYNq(E@BEeWAGwOMhjmYzJ7D{$2tEJCu3necal68VhER"
    "<|}r51|74-;l|cz3Mgi*10)GgG_?)^SV;k!b~0TL4P#g-2xjIXn|AQ}jWbvBuSxe%@+P?M%m"
    "4q@^jA0Zw{?TS`<@ov1ttVslm&An++tHYmS2D?mIcX{h1|N6Mm6X@*vBud?1`v>BVdezKmeF"
    "pywsWTeJeBI|?zjVE$>o0QJ^$U2@KXnCMMU5pKjBOrhUq0uJsju1rESU-wzy7=y+Mk|WV+-1"
    "d~RkN8uK){=O2|+D63tR+~*luK!K)SB`H|UXCsiFo0c;p%cI&tI%;SB>8B9nJ$#_|wKIV@U`"
    "KWF|rNgCW(yr8?nIwpxOm-74>>P5*aS}yfRtQ##R(0VeTCAa#+nEZn-cFo#+@ofQoL`&=g$B"
    "6q`v`%<d5sSKp`F7p{XjH;fDUxm7Yy}wUDA>!wjCe|YBQ2itrc%vj?Ax3XMbABQkdt;UT}GB"
    "lvqg5gHijOzWX=#BTQZYmGNhQ3Q6^=kE$OfN@6q~HPX>&O3dPc?xcyA&+mc07yHbInvb%l!R"
    "#28_NZjY6wU>OdKepw%-4OoT3@wx0%Phn%lR>6z5oWfY%n4amREAMC%#&>{yE1Fq2G(SasYb"
    "2v61#m@hDRZ9(yk3^=}%S%i8-Z?imTirqB#xCC>d*x(nbmHb*&(pg@`Z*%>j3r8lb{|ZmtK@"
    "5Ymh%IwYc^?)gizW1;lNu`dRz)4z@p`zwZ>1LL<|m~J?L2G*IGinjzu5V;2;?&sq?YM?0@EH"
    "S>{dPS_{lwKomq1MTXrIU>vri80_PJo?UWzlSrCI}cYl42PG=t&r`B7He=+q@+r;DY7;I^9K"
    "O9i~U4Vyp#aRo!4xPZf|?Nk1{Abgk=nb8E{iBzkq5y`z?}Ah*FxO-21_u~|{!v%IeUcNcp%#"
    "@<Ywf71n_S)a1*WY8-LxT0)N<x#x3G9*&fWZ2Z^GpXj;>1o6SlnpRy8HP18jgn}msv?tW5*E"
    "&*&8{i4CN^X%^c@@0Sq>E@<1ldn*YIowC8shb&JR3LgJ6NaAAnJI(JB&V;6sreL!Rd8!!$lF"
    "R?t{1?kjb)&O~{@l0aPTG4u9U=c`q~(2hL2^vIJV_b9uUj6}Gr*zVYCWdjsyZK~WZuip;Za^"
    "<`;44R}DN?`&JK1C*hY|t*{GL_<knXM;VGA*SHBQP5QKWV<a8ZmEx-p=JWgRa%H!L8`}<yv9"
    "zXV#hqd>T?T%ZT+(S+0n*d|}kNOtTvT(B*<Jvj4ICZA$M3?$bBa>(%HzHzlB@e%Tbgxgg855"
    "Q0x;pw`Uh<xA#oFod1+t<?@(i&e3O_iQep%4L0-JOrY$3@DSyxScnsH@Sm)Q#E{Sjl?Jxw?f"
    "g0#jjm+CM=hMCTC$Oanx-I2wt-8tcq&0aFNA9M%Gi;v!>BZMK`M^_c>AJZP;Kj<dyk<ax7>m"
    "h%KKlY-(lz<!lvn$3^9GTe<;WS2GbpQqVn?fZ7@@;pOHGN;FWMijraTvvD$OJlbEWIH*FtH<"
    "Ybt*I`nd(&>TnU5|8(Jsc<pwwf7~lkI*7dZ^OPVKC+KUrifq#BNpsc3MyIQVvREbiy}LKo7$"
    "&<1bK|En%MT@?nwJ{(Cr3k7y-pprx3g6(?FVM@&QQq=3;5K!}WC1w=7AREr)nu?8e4IKzuo{"
    "q7t5sIVVg6UD?$FZ!mq_KTQ%G_tgYOEZk8@X<`(hW;$}?ptpatqX4^Q9^?j1*&2dBTCk=f6`"
    "d5a)RVr@0R%C9E2$<=|D#l;Owi1EU|MNNG;!NZhUXq{PO~JTi1H7|1(tJsIdn2*0BGAxZYxJ"
    "c4w*v{cw1Cb~-*B9(&DpyXT!>oL!wAo*nDXJ-fSt!R}%#jzze;ir1@K5b@WUAsP*SxH$WCdg"
    "NVQe7gGR9m02E(R-~KxNKp~B3#={JrXJF*QP`nw9KBlaVn)OZZywvy?M4vmO@thw!K&>su2x"
    "0AI>Et?S+?d*v_q&8if#C*WMi}!@o9aPO|}jTKK2kXk6QB$ipG6;=W(cY3``w-S_^Cemi8}t"
    "d*JP^qQif+0tf%kvPL>Dbh|Q2OkegvR$L{<?!VEc*F?UR+}vgk1mV6HnK~>){^vMGvXDa-Mr"
    "pjPOxI@4CzyxZCcxJjD9fS?(y)8H$FWY{cdz-HS98RIt!DuD?4JxN0*tnxJp2N5j0=-4Wk^I"
    "6StLPr@KUksIBGI?JVEs{%NPQ*4|Z}{N2nY#(`Y9Hl2KlwCh<-ywu>R%0$JlogWz0L=9R-0#"
    "!y6nF+)pid}tJqtNnq8$;Rha@G*EYWtcDAg*kzwbE;t&6hp-aOqKR<>igOEj@$1<gey(i#eA"
    "|Z{!X|t5&7?7p~F&cKma_NY+b?be)B%VTS1qqhL)>IP8j1{LH^wbokw(L0;NrV0vLF?g}$Ee"
    "x*VLOCrq13g-Mrz`I4v;4=E1^w?q3MLcRqdny%6xNJOO%YZU5g_XEGFx41_h_fCRPZ*x1Ix+"
    "2)(1f}&kOnS)JfUgDa6T8BkinwMwYFE@DRdLgLPs^7fc6|6Rzr`7FKhI&fF2?--W@)AHY|ei"
    "(g<)o4}gq#rvqb+{av9l9yMxXf?HLiX8M+{8&mw#%cWPKST<kot12s1<hMk45D9FaXpbmzR%"
    "vi}nd5=qIaI5(Kiv9ebO%i2wk_LJ*2)VI%3P6YJh+s^B#Ti1*jE)6_AN%`$Xjtz=D`0#Ci)8"
    "d0XNFp#Ddew-N+7AT;*~?K>+_MCWy28nv73}zkug$VV)z@aw*1zfOH_fC<<9d%?JE5qa~%rM"
    "-_Ri#GpvD90k;Nvtb&3*mU&&L(Dxk1zXIRg8(&AOAvb73MjpskeE8oKr0|iWBMOd;oV9v8Q}"
    "FdL{X86EEm!)gy$7if=|5rLi4SDrRoA3a<5fmhK)AIs%b=;p|hl2!T-Ly@SF&Z8zME9y3w`="
    "*DegQz3DbQC!AyHi}L^Bb|G8d^A}qX$;-Xhz3b2GRKGJ9jx$#Crvn-Cuh^tzp<chLSKSkWHY"
    "`r`eW4D*ec9cSM}2`%ljkpS?ZcY=gDfzi+GC~JsyPi1%PM{T*~8B#e(9v;<>66jWO6B?rtt3"
    "8^VJNa{ZwQ)mVL{0U*OOEILXRTY=c1EVjJYq*0mYjg2n-Y9lKR;HtLOEgLSjjcWs4$gN8nqY"
    "_{r|68O&Z??GTi8wJf@Q4O>trJqho=|?*T@-gPhK3a6atN`>9(TDRavAMoT<Vl4c3%~^u4+|"
    "8&^y$6MjQSVo@f6jUx)r){q*4nSM=SLe#a?Hl73@wPhnLurHhfc9JkD)&7u}EriPnkp;rvs="
    "(x2G!?(mhRkNPp8@^nR8ch#rAaFi-VbF%QzyJ_iUkOq+#B#7k2qLrsjWoXU7DAs}Hji{_KFY"
    "2m9DZZnqY(1X>UkRl@2d!D#ds+Me)uU+tfJpe-xURBj_Victx?0UF;j|*po~FU`{~H9R^eGY"
    "oq?plGHWC=_HED&anFHW|(Xz!DfF3Tg4DVxGlzd;v{$weI_kq~yq1-9RPL|n-M~OBb6<wk)j"
    "7Y*&T};WuyU-n`v{CR}go&1uQ-=5xSuf9rhrbO!j4od=#b-cMCnv}LM5aIFSIh~3O@U?^D-^"
    "SRKCHZ{NiQ_l$qG(j(K0KH;DLdO29(i?s8Yr_4?kUAot=1R=ibTL(I+@<Sww{l=F(nWUF(s)"
    "C_j^|j>EIVqKb}Yj7%Kv#8*}1#rkFuPBRqQM&dK_*#57!P_ji3C=AObj-dyM48-++tGl+|rm"
    "-OW&aa62WUrhc@rdMy(gvuYNd+k@get4s2AbWriDGl<uK4eJ=3Hmy%&|`jU9@kl9iRJk=9}*"
    "m$hZbbBn0V$aC54WO#rh))uh@xb5NxDZ?fqP?oVN{JAH+{9U%l+8aHNzo{eFFz{I@4;8!5T!"
    "fo(D5n`MGp3M7#wi+UO?s3MtAky$-96ug9m4w`sJCe!cP+*r?Ov{JJWZX!JQE+)kIG=!kgB!"
    "rYY5N1H`~?tOLu%??&@!1zaFvoSmsdO7igm)^McbmL|FT%aw9Prx4Wkig4N%>f>IK6pP+ADS"
    "TG`V%h@10pN7}$fwh4{qit*9cgQCJct8{jQWQT;tRFlR--5VU5(!l)*v7s>_1<q2U3hmwxRh"
    "P^44%awM>i)2GW}2_g$x3tv{IVObx25OS!E0<LpCZ}=>_YnW+^nRY@xBz5Iy9{Ra?|kV$R5f"
    "T7a$3*JzY4lj;efXJf*T6Ql5O<BpUuF!3pTt)GMvB%Xu)Kw|H@{$(b~!hqBq${6UqpB=;Goc"
    "{}O`lpB+GRm~s7pk=*S)Z{;AmfjiS_9!(4<90|+yInDFFx?U|E&|*wYL+3~+YZ7q5ud4JA@^"
    "Pd?+<)vS($t>nW0L^8Zg)Ew$pxvr20$PDAgyyaot0Y-e`8`d$_M7(rbIO$hcIPPZ<Stoyq@9"
    "?h1s}5rGju?pe_?wvc@fLYw!LcZ?Kbe~rGQbXA$HyXSyLp+z;?iK3QcO2YAn6``d()6#>95V"
    "v9V?8Ln~1`yQoqu$LYWK^SAyq%Y69)*^*X7|My{M6+;8p&({XA+s!D2M?XRB*1WFHWwve??K"
    "w|Duz5Rl`aJwXH+NcOL^R>6{@zTIzv1KP_CyM#24{NXF_$Pq10wBN1h!k`kr#n6|1!I3JS)%"
    "D>gXA}#(5*rlhthp(Ap((o^9yc|187)-1dOT&!E_Lxhf#WI4F#$zolL<{8a1++{W|HPBm56R"
    "EemN~XUJVLTzLnV=Au44?45Swcf<jnl2M>Y)E_|(VsT#_!2OOarF_5^AXQ*?hR63LbP1Z#1X"
    "5*?!*Vw2^!4qR^^RLeU)ejLz9n7(y#<(bch);BjB`&FC`Ne*RW?7lafI92O6REdIY5>;e2Ke"
    "5XKu=c|M`ocsh>13oBKgd>F=rx(6I*RB&fZwszNk!-0?I(?A*Kb?<)=I@U01v#p1c1lQ`mbf"
    "vUTjwzun1dtjXs1oLEfz3i3702PU@nu#x3&D`k9-}>bAL3r*$~aHr-m?gVvir&M$z6(7=?zq"
    "iK!U*5@-iG*i36(y`TO;i-OuESzeOMS%PvycJQ3D<9#_rNwLt0dN%(EyABp-oeJvSPL`6hOR"
    "!|nFDCBBl_J7epdMGKA5uj7#k-pb5@NKZ^|^cz>q8*EwCKvgloh5l_u?fTaCgPvPydQYm3?b"
    ";^^J9d3HKGd_BcYA-|AXLaP$hg0>%%wG*2ztp;evBb75$6|{p<-*Y3Ij3+}*iy&%7c|XkXVF"
    "l+0LOUum*pLgFm>%0fjYmza^figf_9c4<=lR&F4}$M#SYS}<g?@EGnI$=`FG%}Ado)0@_536"
    "FT#d8#@3y-vXcE=v_xYPwZ%_6QrhG^fElEwb(Z+~EMh2Q90gcH&1L-FnTIOif_@+km0BTOOt"
    "pJ;)s+W@Brk^CRY*;C=!+o{NO&28c51Yrd)A&B>ovB1<r)4Te%5^lo0g6$}+k#Z?Xm)Zc1Pn"
    "5WrIBm6xH`Z61X6Np@^~FN5k_RQIG7)wsK1`6h34e_;^lnS%xACPHwQ2GXD<-+Awh@wo%^(8"
    "GF9{yCr@>|tmc()L($w5To~#ApM)J`;_T?ok$)B~)}cIlmDzjT-E1CvRr2wdC*Q_!5FeW!8k"
    ";YjfIQ{mFowzoys_#SWiTvZv9WooAF6w3fr?uBjj0UGlZ^pZXnxamJF#F2lb@PeU+hfM@dRo"
    "w>y{q80dI?U)7iV`&3wM7d}<|+$<W(i_%{Y?xGzH!>72F2J!Fsv+OR9}s)JTH&?3_|DnatJK"
    "((uIacv4$pS#ud1QzLTyIdBu?MX_&4!dm|$u$m`AKytTm*vHJYZ1P#=dj>3#5WlD4S^g?{#3"
    "_ey_5fTq7Hf|>vr|cY*ldI(`kO#gZF!9r9PK-EfN;k-cc_W%7$-cnq3GsxNEaabnAE`<8t*n"
    "tvOigV!7#%jc#<_8bWc&$A(OKk0gMxZ9LLs%T^tPmzTDHPe=ctmmFSa&b>N&eDlIhgFzbjxo"
    "n`##A?KN4H#Enw-7UA${j{Ve+{Hjvm><+8|F_6G*VHj$W<>MXILQ;A`Rxi0=v8$utwhUtn62"
    "Hz(d>%N>!ZWP5XHCU8<mK2=`RJIh`$zj;D>?irya2UrmL>l0H?`&N$-EFn%Z2mo|2c7z9^{D"
    "6(?5;vdfsr>}EMP4QSiwa_$rgaLi_x;0>5{NC26z0Cuqv;k?dx$7fjxLRIsz^wcEy0d#nXco"
    "Aq<E6W-h}8>BM({Z%k4eHslDNJYF5;~Zf;G(nXi>2z>WHj%-+p_jo$$69J$BZ{&uZ%^$0xfi"
    "Ll3yZPYhXHXebsGLv`Hr_#l_`w7h%jb9!a8YH(Sq@BIbx_i>u!fEau-%^(ARFrPg?dST_?sX"
    "{z?#RGMy<w=UW?3Fml2#1wX3fYh?GZSiY#bCok9`7-h)!Be3^ef;*7m-|&;0M9>BL;K|a#4W"
    "9#<06U>6LED97g&7dz-pA%?H;R%#hU^X-S4$buVlg#C+)m0e#6F>DjP;qZ5h!MGl-WM4z5GM"
    "`gx6YhbIW<tDl9<PM?lZMg+UTZs|jL&BKp@h&(m2S(oR*u1z{l7mrlGN!BlAn-h4z%WN+p0k"
    "lT=0rzh+#P+ZiV!Ct53jm!#KohnP7Oe7RA?s`S8!j5tQ=qtPz$a(p|LZU$4^BUXBkK2(;9ug"
    "b?4RVJI`*dMMKnRYqY(yY4ncJG+JG`u^WL&`0ZWWEpOBX3#{X-TX1yz3n5Tpy#"
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _live_source_c() -> str:
    source = zlib.decompress(base64.b85decode(_LIVE_SOURCE_C_B85)).decode("utf-8")
    assert _digest(source) == module.LIVE_SOURCE_C_SHA256
    return source


_STAGING_ANCHOR_SHA256 = {
    "seal_evidence_security_imports": (
        "889ec8d49af5c2e1d249f72eedcc32b4fb6c24f02b887310f5264122bba66eca",
        "588a80e6f914e81ee048545fe88db993e53eca956e62197c9e111b6ae5fe9b7e",
    ),
    "declare_controlled_evidence_contract": (
        "306f7b408f6efd7e403a4b5ebe872b45eb3b3b49a72b2faf931fb971f0b5d447",
        "3e1f0a7471fdb48ae06edef8a12ba84581caf81ca7a2ae57576c794d65f59069",
    ),
    "declare_controlled_evidence_receipts": (
        "07988b6d74e6bd9bfbcc3066ce90fb5dc5d468bbeb8dbff206e9d13a4d5c0b1d",
        "3266214c82fdcafa23110b6f8b54d0a025f6574f236639840d397bbc33741b43",
    ),
    "stage_and_verify_controlled_evidence": (
        "ee23596986ff04cebc0b47588287a52c9eede1a1af77614613a9f16c79911682",
        "771568835c7b5470dd95225c9f6827c2b479dcc618e38130486d3d40e590b2f2",
    ),
    "stage_after_controlled_runner": (
        "3b98b9c20127c07ce891865ce70cdceff35e6f2db00291de33dd5d0c3513eff1",
        "990bcccf62a8144ede9e5a9709675822f44fa3eaefed3a2cd74dbac1149bf7e4",
    ),
    "upload_only_sealed_controlled_evidence": (
        "a2b24864626861a0dcf0a97fff24835d7048c8d2d7c9a18d6947305da4f82f5d",
        "5ff96382a57683dffc18e45faf5fba7f9608ba2bb806f3a9565c74503308ff80",
    ),
}


def _anchor(name: str):
    matches = [anchor for anchor in module._ANCHORS if anchor.name == name]
    assert len(matches) == 1
    return matches[0]


def _source_c_fixture() -> str:
    """Small executable source carrying every exact live source-C anchor."""

    source = textwrap.dedent(
        """\
        from __future__ import annotations

        import argparse
        import hashlib
        import json
        from pathlib import Path
        from typing import Mapping, NamedTuple, Sequence

        CANONICAL_1337_PROTOCOL_ID = "DAIR-CAUSAL-1337-v1"
        CANONICAL_1337_SAMPLE_COUNT = 1337
        EXPERIMENT_MAX_EPOCHS = 50
        RTX5090_TRAIN_BATCH_SIZE_PER_GPU = 2
        RTX5090_EVAL_BATCH_SIZE_PER_GPU = 4
        RTX5090_VAL_INTERVAL = 10
        EXPERIMENT_CHECKPOINT_CFG_OPTIONS = ()
        RTX5090_HEADLESS_CFG_OPTIONS = ()


        def _validate_rtx5090_runtime_contract_multi_gpu(contract):
            return contract


        _portable_runtime_validator = _validate_rtx5090_runtime_contract_multi_gpu


        class ExperimentSpec(NamedTuple):
            name: str
            kind: str
            config: str | None
            requires_teacher: bool


        def _positive_integer(value: str) -> int:
            parsed = int(value)
            if parsed <= 0:
                raise argparse.ArgumentTypeError("value must be positive")
            return parsed


        def _sha256_argument(value: str) -> str:
            return value


        def _parser() -> argparse.ArgumentParser:
            parser = argparse.ArgumentParser()
            parser.add_argument("--max-epochs", type=_positive_integer, default=50)
            parser.add_argument("--teacher-checkpoint", type=Path)
            return parser


        def _validate_arguments(args: argparse.Namespace) -> None:
            for field in ("source_dataset_id", "training_dataset_id"):
                value = getattr(args, field)
                if type(value) is not str or not value.strip():
                    raise ValueError(field)


        def _ddp_training_command(
            python: Path,
            *,
            gpus: int,
            config: Path,
            work_dir: Path,
            max_epochs: int,
        ) -> list[str]:
            return [
                str(python),
                "-m",
                "torch.distributed.run",
                "--standalone",
                f"--nproc_per_node={gpus}",
                "--module",
                "tools.resilient_v2x.run_deterministic",
                "tools/train.py",
                str(config),
                "--work-dir",
                str(work_dir),
                "--launcher",
                "pytorch",
                "--cfg-options",
                f"train_cfg.max_epochs={max_epochs}",
                f"train_cfg.val_interval={RTX5090_VAL_INTERVAL}",
                f"train_dataloader.batch_size={RTX5090_TRAIN_BATCH_SIZE_PER_GPU}",
                f"val_dataloader.batch_size={RTX5090_EVAL_BATCH_SIZE_PER_GPU}",
                f"test_dataloader.batch_size={RTX5090_EVAL_BATCH_SIZE_PER_GPU}",
                "find_unused_parameters=True",
                *EXPERIMENT_CHECKPOINT_CFG_OPTIONS,
                *RTX5090_HEADLESS_CFG_OPTIONS,
            ]


        def _baseline_plan_command(
            python: Path,
            *,
            source_root: Path,
            baseline: str,
            training_index: Path,
            work_dir: Path,
        ) -> list[str]:
            return [
                str(python),
                str(source_root / "tools/resilient_v2x/train_controlled_baseline.py"),
                "--baseline",
                baseline,
                "--training-index",
                str(training_index),
                "--work-dir",
                str(work_dir),
                "--seed",
                "20250218",
                "--dry-run",
            ]


        def _read_json_object(path: Path) -> dict[str, object]:
            value = json.loads(path.read_text(encoding="utf-8"))
            assert isinstance(value, dict)
            return value


        def _sha256(path: Path) -> str:
            return hashlib.sha256(path.read_bytes()).hexdigest()


        def _validate_baseline_dry_run(
            *,
            spec: ExperimentSpec,
            work_dir: Path,
        ) -> tuple[Path, Path, dict[str, object]]:
            plan_path = work_dir / "training_plan.json"
            resolved_path = work_dir / "resolved_config.py"
            plan: dict[str, object] = {}
            expected = {
                "schema_version": 1,
                "plan_type": "resilient_v2x_controlled_baseline_training",
                "baseline": spec.name,
                "work_dir": str(work_dir.resolve(strict=True)),
                "plan_path": str(plan_path),
                "resolved_config": str(resolved_path),
                "resume": False,
            }
            plan.update(expected)
            return plan_path, resolved_path, plan


        def _experiment_run_contract(
            args: argparse.Namespace,
            *,
            spec: ExperimentSpec,
            dataset_root: Path,
            config_path: Path,
            training_command: Sequence[str],
            baseline_plan: Mapping[str, object] | None,
        ) -> dict[str, object]:
            resolved_config_sha256 = _sha256(config_path)
            if baseline_plan is None:
                declared_config = str(config_path)
                declared_config_sha256 = resolved_config_sha256
            else:
                declared_config = str(baseline_plan["baseline_config"])
                declared_config_sha256 = str(baseline_plan["baseline_config_sha256"])
            return {
                "schema_version": 1,
                "experiment": spec.name,
                "declared_config": declared_config,
                "declared_config_sha256": declared_config_sha256,
                "max_epochs": EXPERIMENT_MAX_EPOCHS,
                "seed": 20250218,
                "learning_rate": 0.0001,
                "training_command": list(training_command),
                "dataset_root": str(dataset_root),
            }


        def _execute_experiment_from_task(
            args: argparse.Namespace,
            *,
            python: Path,
            source_root: Path,
            dataset_root: Path,
            spec: ExperimentSpec,
            work_dir: Path,
            config_path: Path,
        ) -> tuple[list[str], dict[str, object]]:
            baseline_plan = None
            if spec.kind == "baseline":
                training_index = dataset_root / "protocols/dair_v2/training_overlays.json"
                plan_command = _baseline_plan_command(
                    python,
                    source_root=source_root,
                    baseline=spec.name,
                    training_index=training_index,
                    work_dir=work_dir,
                )
                plan_path, config_path, baseline_plan = _validate_baseline_dry_run(
                    spec=spec,
                    work_dir=work_dir,
                )
                assert plan_command and plan_path
            training_command = _ddp_training_command(
                python,
                gpus=args.gpus,
                config=config_path,
                work_dir=work_dir,
                max_epochs=args.max_epochs,
            )
            contract = _experiment_run_contract(
                args,
                spec=spec,
                dataset_root=dataset_root,
                config_path=config_path,
                training_command=training_command,
                baseline_plan=baseline_plan,
            )
            return training_command, contract


        def _runner_command(
            args: argparse.Namespace,
            *,
            python: Path,
            runner: Path,
        ) -> list[str]:
            command = [
                str(python),
                str(runner),
                "--max-epochs",
                str(args.max_epochs),
            ]
            return command
        """
    )
    imports = _anchor("seal_evidence_security_imports")
    source = source.replace(
        "import argparse\nimport hashlib\nimport json\n"
        "from pathlib import Path\n"
        "from typing import Mapping, NamedTuple, Sequence\n",
        "import argparse\nimport hashlib\nimport json\n"
        + imports.before
        + "from typing import Mapping, NamedTuple, Sequence\n",
        1,
    )
    contract = _anchor("declare_controlled_evidence_contract")
    source = source.replace(
        "CANONICAL_1337_SAMPLE_COUNT = 1337\n",
        "CANONICAL_1337_SAMPLE_COUNT = 1337\n" + contract.before + '    "0" * 64\n)\n',
        1,
    )
    receipts = _anchor("declare_controlled_evidence_receipts")
    source = source.replace(
        textwrap.dedent(
            """\
            class ExperimentSpec(NamedTuple):
                name: str
                kind: str
                config: str | None
                requires_teacher: bool


            """
        ),
        receipts.before + ")\n\n\n",
        1,
    )
    staging = _anchor("stage_and_verify_controlled_evidence")
    post_runner = _anchor("stage_after_controlled_runner")
    sealed_upload = _anchor("upload_only_sealed_controlled_evidence")
    controlled_validation = (
        staging.before
        + "args: argparse.Namespace, *, work_dir: Path, task: object) -> int:\n"
        + "    command: list[str] = []\n"
        + "    source_root = Path('.')\n"
        + "    env: dict[str, str] = {}\n"
        + post_runner.before
        + "}\n"
        + "    runs: list[dict[str, object]] = []\n"
        + sealed_upload.before
        + "\n\n"
    )
    return (
        source
        + "# existing ClearML download hardening\n"
        + "# extract_archive=False\n"
        + "# force_download=True\n"
        + controlled_validation
    )


def _build_fixture(monkeypatch: pytest.MonkeyPatch):
    source_c = _source_c_fixture()
    monkeypatch.setattr(module, "EXPECTED_SOURCE_C_SHA256", _digest(source_c))
    return source_c, module.build_source_d(source_c)


def _namespace(source_d: str) -> dict[str, object]:
    namespace: dict[str, object] = {"__name__": "formal_source_d_fixture"}
    exec(compile(source_d, "<source-d-fixture>", "exec"), namespace)
    return namespace


def _staging_namespace(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    _source_c, result = _build_fixture(monkeypatch)
    namespace = _namespace(result.source_d_text)
    namespace["os"] = os
    return namespace


def _write_evidence_zip(
    namespace: dict[str, object],
    tmp_path: Path,
    *,
    extra_names: tuple[str, ...] = (),
) -> tuple[Path, object]:
    payload = b"sealed controlled evidence"
    names = tuple(f"member-{index:02d}.bin" for index in range(38))
    receipt_type = namespace["ControlledEvidenceMemberReceipt"]
    stage_type = namespace["ControlledEvidenceStage"]
    receipts = tuple(
        receipt_type(
            name,
            len(payload),
            hashlib.sha256(payload).hexdigest(),
            (index + 1,),
        )
        for index, name in enumerate(names)
    )
    stage = stage_type(tmp_path / "sealed-stage", (1,), receipts)
    archive_path = tmp_path / "controlled-evidence.zip"
    with zipfile.ZipFile(
        archive_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        allowZip64=False,
    ) as archive:
        for name in (*names, *extra_names):
            member = zipfile.ZipInfo(name)
            member.create_system = 3
            member.external_attr = (stat.S_IFREG | 0o400) << 16
            archive.writestr(
                member,
                payload,
                compress_type=zipfile.ZIP_DEFLATED,
            )
    return archive_path, stage


def _classic_eocd_offset(archive_bytes: bytes | bytearray) -> int:
    offset = len(archive_bytes) - 22
    assert offset >= 0 and archive_bytes[offset : offset + 4] == b"PK\x05\x06"
    return offset


def _forbid_zipfile_construction(
    namespace: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
) -> list[tuple[object, ...]]:
    calls: list[tuple[object, ...]] = []

    def unexpected_zipfile(*args: object, **_kwargs: object) -> object:
        calls.append(args)
        raise AssertionError("ZipFile must not run before raw ZIP preflight")

    monkeypatch.setattr(namespace["zipfile"], "ZipFile", unexpected_zipfile)
    return calls


def test_generated_valid_exact_inventory_zip_survives_raw_preflight(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    namespace = _staging_namespace(monkeypatch)
    archive_path, stage = _write_evidence_zip(namespace, tmp_path)

    namespace["_verify_uploaded_evidence_zip"](archive_path, stage)


def test_generated_entry_bomb_is_rejected_before_zipfile_construction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    namespace = _staging_namespace(monkeypatch)
    _valid_path, stage = _write_evidence_zip(namespace, tmp_path)
    bomb_path = tmp_path / "entry-bomb.zip"
    with zipfile.ZipFile(
        bomb_path,
        "w",
        compression=zipfile.ZIP_STORED,
        allowZip64=False,
    ) as archive:
        for index in range(2_048):
            archive.writestr(f"bomb-{index:04d}.bin", b"x")
    raw = bytearray(bomb_path.read_bytes())
    eocd = _classic_eocd_offset(raw)
    struct.pack_into("<HH", raw, eocd + 8, 38, 38)
    bomb_path.write_bytes(raw)
    central_directory_size = struct.unpack_from("<L", raw, eocd + 12)[0]
    assert (
        central_directory_size
        > namespace["CONTROLLED_EVIDENCE_MAX_CENTRAL_DIRECTORY_BYTES"]
    )
    assert (
        bomb_path.stat().st_size <= namespace["CONTROLLED_EVIDENCE_MAX_ARCHIVE_BYTES"]
    )
    calls = _forbid_zipfile_construction(namespace, monkeypatch)

    with pytest.raises(RuntimeError, match="EOCD/ZIP64 contract drifted"):
        namespace["_verify_uploaded_evidence_zip"](bomb_path, stage)
    assert calls == []


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("fake_eocd", "EOCD/ZIP64 contract drifted"),
        ("zip64_locator", "ZIP64 locator is forbidden"),
        ("multi_disk", "EOCD/ZIP64 contract drifted"),
        ("central_directory_truncation", "EOCD/ZIP64 contract drifted"),
        ("nul_central_name", "central entry 0 is unsafe"),
        ("trailing_central_entry", "central inventory drifted"),
    ),
)
def test_generated_malformed_zip_is_rejected_before_zipfile_construction(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    namespace = _staging_namespace(monkeypatch)
    extra_names = (
        ("trailing-central-entry.bin",)
        if mutation == ("trailing_central_entry")
        else ()
    )
    archive_path, stage = _write_evidence_zip(
        namespace,
        tmp_path,
        extra_names=extra_names,
    )
    raw = bytearray(archive_path.read_bytes())
    eocd = _classic_eocd_offset(raw)
    if mutation == "fake_eocd":
        raw.extend(raw[eocd:])
    elif mutation == "zip64_locator":
        raw[eocd - 20 : eocd - 16] = b"PK\x06\x07"
    elif mutation == "multi_disk":
        struct.pack_into("<H", raw, eocd + 4, 1)
    elif mutation == "central_directory_truncation":
        del raw[eocd - 1]
    elif mutation == "nul_central_name":
        central_directory_offset = struct.unpack_from("<L", raw, eocd + 16)[0]
        raw[central_directory_offset + 46] = 0
    else:
        assert mutation == "trailing_central_entry"
        struct.pack_into("<HH", raw, eocd + 8, 38, 38)
    archive_path.write_bytes(raw)
    calls = _forbid_zipfile_construction(namespace, monkeypatch)

    with pytest.raises(RuntimeError, match=message):
        namespace["_verify_uploaded_evidence_zip"](archive_path, stage)
    assert calls == []


def test_generated_raw_reload_installs_exact_server_snapshot_and_skips_public_reload(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _staging_namespace(monkeypatch)
    task_id = "a" * 32
    snapshot = SimpleNamespace(id=task_id, artifacts={})
    task = SimpleNamespace(
        id=task_id,
        _offline_mode=False,
        _reload_skip_flag=True,
    )
    public_calls: list[bool] = []

    def raw_reload() -> object:
        assert task._reload_skip_flag is False
        return snapshot

    def public_reload() -> None:
        public_calls.append(True)
        raise AssertionError("public reload is not an evidence readback")

    task._reload = raw_reload
    task.reload = public_reload

    namespace["_reload_controlled_evidence_task"](task)

    assert task._data is snapshot
    assert task._reload_skip_flag is True
    assert public_calls == []


@pytest.mark.parametrize("snapshot", (None, False, 0, 1.0, "", b""))
def test_generated_raw_reload_rejects_missing_or_primitive_snapshots(
    monkeypatch: pytest.MonkeyPatch,
    snapshot: object,
) -> None:
    namespace = _staging_namespace(monkeypatch)
    task = SimpleNamespace(
        id="a" * 32,
        _offline_mode=False,
        _reload_skip_flag=True,
        _reload=lambda: snapshot,
    )

    with pytest.raises(RuntimeError, match="server reload returned no snapshot"):
        namespace["_reload_controlled_evidence_task"](task)
    assert task._reload_skip_flag is True


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("wrong_id", "snapshot identity mismatch"),
        ("offline", "offline reload"),
        ("missing_raw", "cannot be server-reloaded"),
        ("raw_error", "server reload failed"),
    ),
)
def test_generated_raw_reload_fails_closed_and_restores_skip_flag(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    message: str,
) -> None:
    namespace = _staging_namespace(monkeypatch)
    task = SimpleNamespace(
        id="a" * 32,
        _offline_mode=mutation == "offline",
        _reload_skip_flag=True,
    )

    def raw_reload() -> object:
        assert task._reload_skip_flag is False
        if mutation == "raw_error":
            raise OSError("backend unavailable")
        snapshot_id = "b" * 32 if mutation == "wrong_id" else task.id
        return SimpleNamespace(id=snapshot_id, artifacts={})

    if mutation != "missing_raw":
        task._reload = raw_reload

    with pytest.raises(RuntimeError, match=message):
        namespace["_reload_controlled_evidence_task"](task)
    assert task._reload_skip_flag is True


def test_generated_upload_uses_server_empty_snapshot_not_perfect_local_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    namespace = _staging_namespace(monkeypatch)
    namespace["_verify_controlled_evidence_stage"] = lambda _stage: ({}, {})
    task_id = "a" * 32
    required = {
        "run_contract",
        "evaluation_plan",
        "controlled_baseline_metrics",
        "controlled_baseline_evidence",
    }

    class CachedTask:
        def __init__(self) -> None:
            self._data = SimpleNamespace(
                id=task_id,
                artifacts={name: object() for name in required},
            )
            self._offline_mode = False
            self._reload_skip_flag = True

        @property
        def id(self) -> str:
            return self._data.id

        @property
        def artifacts(self) -> object:
            return self._data.artifacts

        def upload_artifact(self, *_args: object, **_kwargs: object) -> bool:
            return True

        def flush(self, *, wait_for_uploads: bool) -> None:
            assert wait_for_uploads is True

        def _reload(self) -> object:
            assert self._reload_skip_flag is False
            return SimpleNamespace(id=task_id, artifacts={})

        def reload(self) -> None:
            raise AssertionError("public reload must never be called")

    task = CachedTask()
    with pytest.raises(RuntimeError, match="exact artifact inventory drifted"):
        namespace["_upload_controlled_baseline_artifacts"](
            task,
            evidence_stage=SimpleNamespace(root=Path("/sealed-stage")),
        )
    assert task.artifacts == {}
    assert task._reload_skip_flag is True


@pytest.mark.parametrize("flush_result", (False, 0))
def test_generated_upload_rejects_false_or_zero_flush_before_reload(
    monkeypatch: pytest.MonkeyPatch,
    flush_result: object,
) -> None:
    namespace = _staging_namespace(monkeypatch)
    namespace["_verify_controlled_evidence_stage"] = lambda _stage: ({}, {})
    task = SimpleNamespace(id="a" * 32)
    task.upload_artifact = lambda *_args, **_kwargs: True
    task.flush = lambda **_kwargs: flush_result

    def unexpected_reload() -> object:
        raise AssertionError("raw reload must wait for a successful flush")

    task._reload = unexpected_reload

    with pytest.raises(
        RuntimeError,
        match="failed to flush controlled baseline artifact uploads",
    ):
        namespace["_upload_controlled_baseline_artifacts"](
            task,
            evidence_stage=SimpleNamespace(root=Path("/sealed-stage")),
        )


def _dataset_root(tmp_path: Path) -> Path:
    root = tmp_path / "dataset"
    protocol_dir = root / "protocols/dair_v2"
    protocol_dir.mkdir(parents=True)
    (protocol_dir / "training_overlays.json").write_text(
        json.dumps({"protocol_seed": module.TRAINING_OVERLAY_PROTOCOL_SEED}),
        encoding="utf-8",
    )
    return root


def test_live_source_c_identity_is_pinned() -> None:
    assert module.EXPECTED_SOURCE_C_SHA256 == (
        "fbd8ddb4294674ce4a9ecac38bd5f48808e754462cc96df21db6330b60a6a66f"
    )
    assert module.LIVE_SOURCE_C_SHA256 == module.EXPECTED_SOURCE_C_SHA256
    assert module.EXPECTED_SOURCE_D_SHA256 == (
        "e7a9ab0fb05339223cf2c18c52eb72652c311733bf96d1058aa7a769096cf8c3"
    )
    assert module.EXPECTED_EQUIVALENCE_ARTIFACT_SHA256 == (
        "1156fe53f2fe924f91c1c6b50b6b21090d98cd2d74840d6d6f5a358316420433"
    )
    assert module.TRANSFORMATION_ID == (
        "source-c-to-source-d-explicit-seed-evidence-v2"
    )
    assert module.EXPECTED_DECLARED_REPLACEMENT_COUNT == 22
    assert module.EXPECTED_UNCHANGED_SEGMENT_COUNT == 23
    assert hashlib.sha256(MODULE_PATH.read_bytes()).hexdigest() == (
        "904fafd08d710b03b62bc57140121f2a546ada8fa2fb8763e6db8a44b9f8f7e7"
    )


def test_live_fbd8_recomputes_exact_v2_source_d_and_equivalence() -> None:
    source_c = _live_source_c()
    result = module.build_source_d(source_c)

    assert len(source_c.encode("utf-8")) == 137_249
    assert result.source_d_sha256 == module.EXPECTED_SOURCE_D_SHA256
    assert len(result.source_d_text.encode("utf-8")) == 192_238
    assert len(result.source_d_text.splitlines()) == 5_167
    assert result.artifact["artifact_sha256"] == (
        module.EXPECTED_EQUIVALENCE_ARTIFACT_SHA256
    )
    assert [entry["name"] for entry in result.artifact["diff"]] == [
        anchor.name
        for anchor in sorted(
            module._ANCHORS, key=lambda item: source_c.index(item.before)
        )
    ]
    equivalence = result.artifact["equivalence"]
    assert equivalence["declared_replacement_count"] == 22
    assert equivalence["unchanged_segment_count"] == 23
    assert equivalence["unchanged_size_bytes"] == 133_877
    assert equivalence["source_c_replay_sha256"] == module.LIVE_SOURCE_C_SHA256
    assert equivalence["source_d_replay_sha256"] == result.source_d_sha256
    for fragment, expected_count in module._STAGING_REQUIRED_FRAGMENTS:
        assert result.source_d_text.count(fragment) == expected_count
    for fragment in module._STAGING_FORBIDDEN_FRAGMENTS:
        assert fragment not in result.source_d_text


def test_staging_anchor_bytes_are_independently_pinned() -> None:
    observed = {
        anchor.name: (_digest(anchor.before), _digest(anchor.after))
        for anchor in module._ANCHORS
        if anchor.name in module._STAGING_ANCHOR_NAMES
    }
    assert tuple(observed) == module._STAGING_ANCHOR_NAMES
    assert observed == _STAGING_ANCHOR_SHA256


def test_build_records_only_declared_replacements_and_replays(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_c, result = _build_fixture(monkeypatch)

    assert result.source_d_sha256 == _digest(result.source_d_text)
    assert result.source_d_text.count("20250218") == 1
    assert result.source_d_text.count(module.PORTABLE_RUNNER_LOAD_MARKER) == (
        module.EXPECTED_PORTABLE_RUNNER_LOAD_MARKER_COUNT
    )
    assert module.LEGACY_RUNNER_LOAD_TARGET_ANCHOR not in result.source_d_text
    assert '"--seed",\n        "20250218"' not in result.source_d_text
    assert '"seed": 20250218' not in result.source_d_text
    assert [record["name"] for record in result.artifact["diff"]] == [
        anchor.name
        for anchor in sorted(
            module._ANCHORS, key=lambda item: source_c.index(item.before)
        )
    ]
    equivalence = result.artifact["equivalence"]
    assert equivalence["only_declared_anchor_replacements"] is True
    assert equivalence["declared_replacement_count"] == len(module._ANCHORS)
    assert equivalence["source_c_replay_sha256"] == _digest(source_c)
    assert equivalence["source_d_replay_sha256"] == result.source_d_sha256
    assert equivalence["source_d_compiles"] is True
    unsealed = dict(result.artifact)
    seal = unsealed.pop("artifact_sha256")
    assert seal == _digest(module._canonical_json(unsealed))
    assert (
        module.verify_source_d(
            source_c,
            result.source_d_text,
            result.artifact,
        )
        == result
    )


@pytest.mark.parametrize("mutation", ("portable_marker", "legacy_target"))
def test_nonportable_source_c_fails_closed_before_transformation(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    source_c = _source_c_fixture()
    if mutation == "portable_marker":
        source_c = source_c.replace(
            module.PORTABLE_RUNNER_LOAD_MARKER,
            "_drifted_runtime_contract_multi_gpu",
            1,
        )
        message = "portable runner-load marker count mismatch"
    else:
        source_c += module.LEGACY_RUNNER_LOAD_TARGET_ANCHOR
        message = "legacy runner-load target anchor"
    monkeypatch.setattr(module, "EXPECTED_SOURCE_C_SHA256", _digest(source_c))

    with pytest.raises(module.SourceDSeedError, match=message):
        module.build_source_d(source_c)


def test_two_training_seeds_change_commands_and_contract_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, result = _build_fixture(monkeypatch)
    namespace = _namespace(result.source_d_text)
    ddp_command = namespace["_ddp_training_command"]
    baseline_command = namespace["_baseline_plan_command"]
    run_contract = namespace["_experiment_run_contract"]
    parser = namespace["_parser"]()

    assert parser.parse_args(["--training-seed", "17"]).training_seed == 17
    assert parser.parse_args(["--training-seed", "29"]).training_seed == 29
    config = tmp_path / "config.py"
    config.write_text("model = {}\n", encoding="utf-8")
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    root = _dataset_root(tmp_path)
    spec = namespace["ExperimentSpec"]("ffnet", "baseline", None, True)

    commands: dict[int, list[str]] = {}
    baseline_commands: dict[int, list[str]] = {}
    contracts: dict[int, dict[str, object]] = {}
    for seed in (17, 29):
        commands[seed] = ddp_command(
            Path("/python"),
            gpus=4,
            config=config,
            work_dir=work_dir,
            max_epochs=50,
            training_seed=seed,
        )
        baseline_commands[seed] = baseline_command(
            Path("/python"),
            source_root=tmp_path,
            baseline="ffnet",
            training_index=(root / "protocols/dair_v2/training_overlays.json"),
            work_dir=work_dir,
            training_seed=seed,
        )
        plan = {
            "baseline_config": str(config),
            "baseline_config_sha256": _digest(config.read_text(encoding="utf-8")),
            "seed": seed,
            "training_index_protocol_seed": (module.TRAINING_OVERLAY_PROTOCOL_SEED),
        }
        contracts[seed] = run_contract(
            SimpleNamespace(training_seed=seed),
            spec=spec,
            dataset_root=root,
            config_path=config,
            training_command=commands[seed],
            baseline_plan=plan,
        )

    assert commands[17] != commands[29]
    for binding in module._TRAINING_CFG_BINDINGS:
        assert f"{binding}=17" in commands[17]
        assert f"{binding}=29" in commands[29]
    assert baseline_commands[17][-3:] == ["--seed", "17", "--dry-run"]
    assert baseline_commands[29][-3:] == ["--seed", "29", "--dry-run"]
    assert contracts[17]["training_seed"] == contracts[17]["seed"] == 17
    assert contracts[29]["training_seed"] == contracts[29]["seed"] == 29
    assert contracts[17]["training_overlay_protocol_seed"] == 20_250_218
    assert contracts[29]["training_overlay_protocol_seed"] == 20_250_218

    normalized = copy.deepcopy(contracts)
    for seed in normalized:
        normalized[seed]["training_seed"] = "<seed>"
        normalized[seed]["seed"] = "<seed>"
        seed_options = {
            f"{binding}={seed}": f"{binding}=<seed>"
            for binding in module._TRAINING_CFG_BINDINGS
        }
        normalized[seed]["training_command"] = [
            seed_options.get(item, item)
            for item in normalized[seed]["training_command"]
        ]
    assert normalized[17] == normalized[29]


def test_boolean_baseline_plan_seed_cannot_alias_zero(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, result = _build_fixture(monkeypatch)
    namespace = _namespace(result.source_d_text)
    config = tmp_path / "config.py"
    config.write_text("model = {}\n", encoding="utf-8")
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    root = _dataset_root(tmp_path)
    spec = namespace["ExperimentSpec"]("ffnet", "baseline", None, True)
    command = namespace["_ddp_training_command"](
        Path("/python"),
        gpus=4,
        config=config,
        work_dir=work_dir,
        max_epochs=50,
        training_seed=0,
    )
    plan = {
        "baseline_config": str(config),
        "baseline_config_sha256": _digest(config.read_text(encoding="utf-8")),
        "seed": False,
        "training_index_protocol_seed": module.TRAINING_OVERLAY_PROTOCOL_SEED,
    }

    with pytest.raises(ValueError, match="baseline plan training seed mismatch"):
        namespace["_experiment_run_contract"](
            SimpleNamespace(training_seed=0),
            spec=spec,
            dataset_root=root,
            config_path=config,
            training_command=command,
            baseline_plan=plan,
        )


def test_float_baseline_plan_protocol_seed_cannot_alias_integer(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, result = _build_fixture(monkeypatch)
    namespace = _namespace(result.source_d_text)
    config = tmp_path / "config.py"
    config.write_text("model = {}\n", encoding="utf-8")
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    root = _dataset_root(tmp_path)
    spec = namespace["ExperimentSpec"]("ffnet", "baseline", None, True)
    command = namespace["_ddp_training_command"](
        Path("/python"),
        gpus=4,
        config=config,
        work_dir=work_dir,
        max_epochs=50,
        training_seed=17,
    )
    plan = {
        "baseline_config": str(config),
        "baseline_config_sha256": _digest(config.read_text(encoding="utf-8")),
        "seed": 17,
        "training_index_protocol_seed": float(module.TRAINING_OVERLAY_PROTOCOL_SEED),
    }

    with pytest.raises(
        ValueError, match="baseline plan training overlay protocol seed mismatch"
    ):
        namespace["_experiment_run_contract"](
            SimpleNamespace(training_seed=17),
            spec=spec,
            dataset_root=root,
            config_path=config,
            training_command=command,
            baseline_plan=plan,
        )


def test_overlay_protocol_seed_is_read_and_rejected_on_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _, result = _build_fixture(monkeypatch)
    namespace = _namespace(result.source_d_text)
    root = _dataset_root(tmp_path)

    assert namespace["_training_overlay_protocol_seed"](root) == 20_250_218
    index = root / "protocols/dair_v2/training_overlays.json"
    index.write_text(json.dumps({"protocol_seed": 7}), encoding="utf-8")
    with pytest.raises(ValueError, match="overlay protocol seed mismatch"):
        namespace["_training_overlay_protocol_seed"](root)
    index.write_text(
        json.dumps({"protocol_seed": float(module.TRAINING_OVERLAY_PROTOCOL_SEED)}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="overlay protocol seed mismatch"):
        namespace["_training_overlay_protocol_seed"](root)


@pytest.mark.parametrize("seed", [False, -1, 2**32])
def test_training_seed_range_fails_closed(seed: object) -> None:
    with pytest.raises(module.SourceDSeedError, match="training seed must be"):
        module.validate_training_seed(seed)


def test_transformed_cli_seed_range_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _, result = _build_fixture(monkeypatch)
    namespace = _namespace(result.source_d_text)
    parse_seed = namespace["_training_seed_argument"]

    assert parse_seed("0") == 0
    assert parse_seed(str(2**32 - 1)) == 2**32 - 1
    with pytest.raises(argparse.ArgumentTypeError, match="value must be"):
        parse_seed("-1")
    with pytest.raises(argparse.ArgumentTypeError, match="value must be"):
        parse_seed(str(2**32))


def test_source_c_hash_and_anchor_drift_are_distinct_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_c = _source_c_fixture()
    monkeypatch.setattr(module, "EXPECTED_SOURCE_C_SHA256", _digest(source_c))
    module.build_source_d(source_c)

    drifted = source_c.replace('        "20250218",', '        "20250219",', 1)
    with pytest.raises(module.SourceDSeedError, match="source-C SHA-256 mismatch"):
        module.build_source_d(drifted)

    monkeypatch.setattr(module, "EXPECTED_SOURCE_C_SHA256", _digest(drifted))
    with pytest.raises(module.SourceDSeedError, match="anchor .* count mismatch"):
        module.build_source_d(drifted)


@pytest.mark.parametrize("anchor_name", module._STAGING_ANCHOR_NAMES)
def test_each_staging_anchor_drift_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    anchor_name: str,
) -> None:
    source_c = _source_c_fixture()
    anchor = _anchor(anchor_name)
    drifted_before = "#" + anchor.before[1:]
    drifted = source_c.replace(anchor.before, drifted_before, 1)
    assert drifted != source_c
    monkeypatch.setattr(module, "EXPECTED_SOURCE_C_SHA256", _digest(drifted))

    with pytest.raises(
        module.SourceDSeedError,
        match=rf"anchor {anchor_name!r} count mismatch",
    ):
        module.build_source_d(drifted)


@pytest.mark.parametrize(
    ("anchor_name", "original", "mutation"),
    (
        (
            "stage_and_verify_controlled_evidence",
            "if len(ordered) != 38 or sum(item.size_bytes for item in ordered) > (",
            "if len(ordered) != 38.0 or sum(item.size_bytes for item in ordered) > (",
        ),
        (
            "stage_and_verify_controlled_evidence",
            "changed during readback",
            "changed after readback",
        ),
        (
            "stage_and_verify_controlled_evidence",
            "st_nlink != 1",
            "st_nlink < 1",
        ),
        (
            "stage_and_verify_controlled_evidence",
            "or entries_on_disk != 38\n",
            "or entries_on_disk != 38.0\n",
        ),
        (
            "stage_and_verify_controlled_evidence",
            'reloader = getattr(task, "_reload", None)',
            'reloader = getattr(task, "reload", None)',
        ),
        (
            "stage_and_verify_controlled_evidence",
            '("controlled_baseline_evidence", str(evidence_stage.root))',
            '("controlled_baseline_evidence", str(work_dir))',
        ),
    ),
)
def test_staging_security_fragment_drift_cannot_be_resealed(
    monkeypatch: pytest.MonkeyPatch,
    anchor_name: str,
    original: str,
    mutation: str,
) -> None:
    anchor = _anchor(anchor_name)
    assert original in anchor.after
    mutated = anchor._replace(after=anchor.after.replace(original, mutation, 1))
    monkeypatch.setattr(
        module,
        "_ANCHORS",
        tuple(
            mutated if item.name == anchor_name else item for item in module._ANCHORS
        ),
    )
    source_c = _source_c_fixture()
    monkeypatch.setattr(module, "EXPECTED_SOURCE_C_SHA256", _digest(source_c))

    with pytest.raises(
        module.SourceDSeedError,
        match="staging invariant count mismatch|unsafe evidence upload",
    ):
        module.build_source_d(source_c)


def test_rehashed_unchanged_segment_replay_drift_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_c, result = _build_fixture(monkeypatch)
    artifact = copy.deepcopy(result.artifact)
    segments = artifact["equivalence"]["unchanged_segments"]
    assert segments
    segments[0]["sha256"] = "0" * 64
    artifact = module._sealed(artifact)

    with pytest.raises(module.SourceDSeedError, match="artifact mismatch"):
        module.verify_source_d(source_c, result.source_d_text, artifact)


def test_unknown_source_d_diff_or_artifact_mutation_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_c, result = _build_fixture(monkeypatch)
    with pytest.raises(module.SourceDSeedError, match="undeclared diff"):
        module.verify_source_d(
            source_c,
            result.source_d_text + "# undeclared\n",
            result.artifact,
        )

    artifact = copy.deepcopy(result.artifact)
    artifact["equivalence"]["only_declared_anchor_replacements"] = False
    with pytest.raises(module.SourceDSeedError, match="artifact(?: SHA-256)? mismatch"):
        module.verify_source_d(source_c, result.source_d_text, artifact)


@pytest.mark.parametrize("mutation", ("schema_version", "nested_expected_count"))
def test_boolean_integer_alias_without_rehash_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    source_c, result = _build_fixture(monkeypatch)
    artifact = copy.deepcopy(result.artifact)
    if mutation == "schema_version":
        assert artifact["schema_version"] == 1
        artifact["schema_version"] = True
    else:
        diff = artifact["diff"]
        assert isinstance(diff, list)
        assert diff[0]["expected_count"] == 1
        diff[0]["expected_count"] = True
    with pytest.raises(module.SourceDSeedError, match="artifact SHA-256 mismatch"):
        module.verify_source_d(source_c, result.source_d_text, artifact)


def test_rehashed_boolean_integer_alias_is_rejected_by_canonical_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source_c, result = _build_fixture(monkeypatch)
    artifact = copy.deepcopy(result.artifact)
    artifact["schema_version"] = True
    artifact = module._sealed(artifact)
    with pytest.raises(module.SourceDSeedError, match="artifact mismatch"):
        module.verify_source_d(source_c, result.source_d_text, artifact)
