# Modeli 2D i palosjes së proteinave Hydrophobic–Polar (HP)

Ky dokument përkufizon një model të thjeshtuar për përfaqësimin dhe optimizimin e strukturave të proteinave në një rrjetë katrore dy-dimensionale. Çdo aminoacid klasifikohet si hidrofobik \(H\) ose polar \(P\). Modeli favorizon kontaktet mes mbetjeve hidrofobike që nuk janë fqinje në sekuencë.

## 1. Klasifikimi i 20 aminoacideve standarde

Përdoret ky konvencion i qartë binar:

| Klasa | Aminoacidet (kodi me një shkronjë) |
|---|---|
| **Hidrofobike (H)** | Alaninë **A**, Cisteinë **C**, Glicinë **G**, Isoleucinë **I**, Leucinë **L**, Metioninë **M**, Fenilalaninë **F**, Prolinë **P**, Triptofan **W**, Tirozinë **Y**, Valinë **V** |
| **Polare (P)** | Argininë **R**, Asparaginë **N**, Acidi aspartik **D**, Glutaminë **Q**, Acidi glutamik **E**, Lizinë **K**, Serinë **S**, Treoninë **T**, Histidinë **H** |

Ky është një konvencion modelimi, jo një kufi biokimik universal. Disa mbetje, si cisteina, glicina, prolina, tirozina dhe histidina, kanë veti që nuk përshkruhen plotësisht nga një ndarje binare. I njëjti konvencion duhet përdorur në të gjitha eksperimentet dhe duhet dokumentuar.

Shkronja **P** në sekuencën e aminoacideve nënkupton prolinën; etiketa **P** në sekuencën HP nënkupton mbetje polare. Konteksti ose emërtimi i ndryshëm i sekuencave i dallon këto dy përdorime.

## 2. Shndërrimi i sekuencës në H/P

Le të jetë sekuenca e aminoacideve

\[
S=(s_1,s_2,\ldots,s_N),
\]

ku çdo \(s_i\) është një nga 20 aminoacidet standarde. Përkufizohet funksioni \(f\):

\[
f(s_i)=
\begin{cases}
H & \text{nëse }s_i\in\{A,C,G,I,L,M,F,P,W,Y,V\},\\
P & \text{nëse }s_i\in\{R,N,D,Q,E,K,S,T,H\}.
\end{cases}
\]

Sekuenca e modelit është

\[
Q=(q_1,q_2,\ldots,q_N), \qquad q_i=f(s_i).
\]

Shndërrումը ruan gjatësinë dhe rendin e sekuencës. Për shembull:

\[
\texttt{A-K-D-V}\longrightarrow\texttt{H-P-P-H}.
\]

Shkronjat e panjohura ose simbolet e paqarta duhet të refuzohen ose të trajtohen sipas një politike të dokumentuar, jo të klasifikohen në heshtje.

## 3. Përfaqësimi në rrjetë dhe drejtimet relative

Përdoret rrjeta katrore \(\mathbb{Z}^2\). Aminoacidi në pozicionin \(i\) vendoset në koordinatën

\[
x_i=(a_i,b_i), \qquad a_i,b_i\in\mathbb{Z}.
\]

Zinxhiri përcaktohet nga koordinata fillestare dhe \(N-1\) lëvizje relative:

\[
d_k\in\{R,L,U,D\}, \qquad 1\le k<N,
\]

ku vektorët e drejtimit janë

\[
v(R)=(1,0),\quad v(L)=(-1,0),\quad
v(U)=(0,1),\quad v(D)=(0,-1).
\]

Koordinatat ndërtohen në mënyrë kumulative:

\[
x_1=(0,0),\qquad x_{k+1}=x_k+v(d_k).
\]

Shembull: për lëvizjet \(R,U,R,D\), koordinatat janë

\[
(0,0),\ (1,0),\ (1,1),\ (2,1),\ (2,0).
\]

Katër lëvizje vendosin pesë mbetje.

## 4. Kufizimet strukturore

### Vetëshmangia

Asnjë çift mbetjesh të ndryshme nuk mund të zërë të njëjtën pikë:

\[
x_i\ne x_j \qquad \text{për çdo }i\ne j.
\]

Kjo ndalon kryqëzimin e zinxhirit dhe mbivendosjen e mbetjeve.

### Lidhshmëria e zinxhirit

Çdo çift mbetjesh të njëpasnjëshme në sekuencë duhet të jetë fqinj në rrjetë. Me distancën Manhattan:

\[
\|x_{i+1}-x_i\|_1
=|a_{i+1}-a_i|+|b_{i+1}-b_i|=1,
\qquad 1\le i<N.
\]

Ndërtimi kumulativ me drejtimet \(R,L,U,D\) garanton lidhshmërinë. Vetëshmangia duhet kontrolluar veçmas.

Për shembull, lëvizjet \(R,U,L\) nga \((0,0)\) japin koordinatat

\[
(0,0),\ (1,0),\ (1,1),\ (0,1),
\]

që formojnë një zinxhir të lidhur dhe vetëshmangës. Lëvizjet \(R,L\) kthehen në pikën fillestare, ndaj shkelin vetëshmangien.

## 5. Kontaktet hidrofobike jo të njëpasnjëshme

Dy mbetje janë në kontakt kur janë fqinje në rrjetë, por jo njëpasnjëshme në zinxhir. Përkufizohet:

\[
C_{ij}(X)=
\begin{cases}
1 & \text{nëse }\|x_i-x_j\|_1=1\text{ dhe }|i-j|>1,\\
0 & \text{përndryshe}.
\end{cases}
\]

Kontakti H–H llogaritet vetëm kur të dyja mbetjet kanë etiketën \(H\). Çdo çift numërohet një herë, për shembull duke marrë vetëm \(i<j\). Lidhjet e zinxhirit mes mbetjeve të njëpasnjëshme nuk numërohen si kontakte shtesë.

Një funksion i zakonshëm energjie është

\[
E(X)=-\epsilon
\sum_{\substack{1\le i<j\le N\\|i-j|>1}}
\mathbf{1}[q_i=H]\mathbf{1}[q_j=H]C_{ij}(X),
\qquad \epsilon>0,
\]

ku \(\mathbf{1}[\cdot]\) është funksioni indikator. Energjia më e ulët nënkupton më shumë kontakte H–H.

**Shembull:** Për sekuencën HP \(\texttt{HPPH}\), lëvizjet \(R,U,L\) japin:

\[
x_1=(0,0),\quad x_2=(1,0),\quad x_3=(1,1),\quad x_4=(0,1).
\]

Mbetjet 1 dhe 4 janë të dyja H, nuk janë njëpasnjëshme në sekuencë dhe janë fqinje në rrjetë:

\[
\|x_1-x_4\|_1=1,\qquad |1-4|>1.
\]

Prandaj formojnë një kontakt H–H. Për \(\epsilon=1\), energjia është \(E=-1\).

## 6. Simetria e rrotullimit dhe pasqyrimit

Zhvendosja e të gjithë strukturës nuk ndryshon formën apo energjinë. Po ashtu, rrotullimi ose pasqyrimi i strukturës ruan fqinjësinë, vetëshmangien dhe numrin e kontakteve H–H. Kështu krijohen paraqitje të ndryshme koordinatash për të njëjtën strukturë fizike.

Për të shmangur kërkimin e përsëritur të strukturave ekuivalente, mund të përdoren këto rregulla simetrie:

1. Vendosni mbetjen e parë në \(x_1=(0,0)\), duke hequr simetrinë e zhvendosjes.
2. Nëse \(N>1\), kërkoni që lëvizja e parë të jetë \(R\), duke fiksuar orientimin fillestar dhe hequr rrotullimet ekuivalente.
3. Për të hequr pasqyrimin e mbetur ndaj boshtit horizontal, kërkoni që kthesa e parë jo e drejtë të jetë \(U\), jo \(D\). Nëse zinxhiri nuk kthehet, rregulli nuk është i nevojshëm.

Këto rregulla heqin dublikimet pa ndryshuar energjinë e modelit. Renditja e sekuencës duhet ruajtur; kthimi i zinxhirit nuk është përgjithësisht ekuivalent për një sekuencë të caktuar.
