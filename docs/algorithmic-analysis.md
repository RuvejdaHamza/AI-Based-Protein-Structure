# Analiza algoritmike e problemit

## AI-Based Protein Structure Prediction and Optimization

Ky dokument përshkruan formulimin algoritmik të një projekti për optimizimin e palosjes së proteinave duke përdorur modelin 2D Hydrophobic-Polar (HP). Qëllimi është të krahasohen Simulated Annealing, Genetic Algorithm dhe një metodë hibride që kombinon të dyja.

## 1. Formulimi matematikor

Le të jetë sekuenca hyrëse e aminoacideve

\[
S=(s_1,s_2,\ldots,s_N),
\]

ku \(N\) është gjatësia e sekuencës. Në modelin HP, çdo aminoacid klasifikohet si hidrofobik \(H\) ose polar \(P\):

\[
h_i =
\begin{cases}
H & \text{nëse mbetja } i \text{ është hidrofobike}\\
P & \text{nëse mbetja } i \text{ është polare}.
\end{cases}
\]

Një konformacion kandidat përfaqësohet nga koordinatat e mbetjeve në rrjetë:

\[
X=(x_1,x_2,\ldots,x_N), \qquad x_i\in\mathbb{Z}^2.
\]

Konformacioni duhet të përmbushë dy kufizime:

1. **Lidhshmëria e zinxhirit:** mbetjet fqinje në sekuencë zënë pika fqinje në rrjetë:
   \[
   \|x_{i+1}-x_i\|_1=1 \quad \text{për } 1\le i<N.
   \]
2. **Vetëshmangia:** dy mbetje të ndryshme nuk mund të zënë të njëjtën pikë:
   \[
   x_i\ne x_j \quad \text{për } i\ne j.
   \]

Kontakt quhet një çift mbetjesh jo fqinje në sekuencë, por fqinje në rrjetë. Le të jetë:

\[
C_{ij}(X)=
\begin{cases}
1 & \text{nëse } \|x_i-x_j\|_1=1 \text{ dhe } |i-j|>1\\
0 & \text{përndryshe}.
\end{cases}
\]

Energjia bazë e modelit HP është:

\[
E(X)=-\epsilon
\sum_{\substack{i<j\\|i-j|>1}}
\mathbf{1}[h_i=H]\mathbf{1}[h_j=H]C_{ij}(X),
\]

ku \(\epsilon>0\) është energjia e një kontakti hidrofobik dhe \(\mathbf{1}[\cdot]\) është funksioni indikator. Energjia bëhet më e ulët kur formohen më shumë kontakte H–H. Problemi është të gjendet një konformacion i vlefshëm \(X\) me energji sa më të ulët.

Kjo është një detyrë optimizimi kombinator me kufizime mbi ecjet vetëshmangëse në rrjetë. Numri i konformacioneve të mundshme rritet në mënyrë eksponenciale me gjatësinë e sekuencës, prandaj kërkimi shterues bëhet shpejt jopraktik.

## 2. Pse është i përshtatshëm modeli HP?

Modeli HP është i dobishëm për një projekt të dizajnit të algoritmeve sepse e thjeshton palosjen e proteinave në një objektiv të qartë dhe të matshëm, duke ruajtur intuitën se mbetjet hidrofobike priren të grumbullohen larg tretësit.

Përparësitë kryesore janë:

- Përfaqësim kompakt: çdo aminoacid klasifikohet vetëm si H ose P.
- Funksion energjie i thjeshtë për t'u llogaritur dhe verifikuar.
- Hapësirë kërkimi diskrete, e përshtatshme për SA, GA dhe algoritme hibride.
- Krahasim i lehtë i konformacioneve dhe i rezultateve të algoritmeve.
- Nuk kërkon infrastrukturë të simulimit molekular apo të dhëna të mëdha biologjike për eksperimentet bazë.

Modeli është një abstraksion për optimizim, jo simulim i hollësishëm fizik i një proteine reale.

## 3. Supozimet dhe kufizimet

Modeli bazë supozon se:

- Proteina paraqitet në një rrjetë katrore dy-dimensionale.
- Çdo mbetje zë një pikë të vetme të rrjetës.
- Tërheqja përfaqësohet nga energji uniforme për kontaktet H–H.
- Mbetjet polare nuk kontribuojnë në tërheqje në versionin më të thjeshtë të modelit.
- Konformacioni është statik; dinamika e palosjes dhe tretësi nuk modelohen drejtpërdrejt.
- Sekuenca dhe klasifikimi H/P janë të sakta.

Kufizimet duhen theksuar në raport:

- Proteinat reale janë tredimensionale; modeli 2D nuk paraqet të gjitha kontaktet e mundshme.
- Dy klasat H/P nuk përfaqësojnë hollësitë e kimisë së aminoacideve, lidhjet hidrogjenore, ngarkesat elektrike apo gjeometrinë e zinxhirëve anësorë.
- Një energji e ulët në model nuk dëshmon se struktura është struktura natyrore, biologjikisht e saktë ose funksionale.
- Mund të ketë shumë konformacione me të njëjtën energji.
- Rezultatet varen nga sekuenca, operatorët e lëvizjes, parametrat dhe fara e rastësisë.

Prandaj, vlefshmëria duhet kuptuar në dy nivele: **vlefshmëria e strukturës në rrjetë** (plotësimi i kufizimeve të modelit) dhe **vlefshmëria biologjike** (e cila nuk garantohet nga modeli HP).

## 4. Objektivi i optimizimit

Objektivi kryesor është minimizimi i energjisë HP mbi të gjitha konformacionet e vlefshme:

\[
\min_{X\in\mathcal{F}} E(X),
\]

ku \(\mathcal{F}\) është bashkësia e konformacioneve që plotësojnë lidhshmërinë dhe vetëshmangien.

Për algoritme që maksimizojnë fitness-in, mund të përdoret \(F(X)=-E(X)\), ose drejtpërdrejt numri i kontakteve H–H. Konformacionet e pavlefshme duhet të refuzohen ose të riparohen. Nëse përdoren penalizime, ato duhet të jenë mjaftueshëm të mëdha që një strukturë e pavlefshme të mos duket më e mirë se një strukturë e vlefshme.

Projekti mat aftësinë e algoritmeve për të gjetur konformacione me energji të ulët në modelin HP; nuk duhet të pretendojë se përcakton strukturën natyrore të proteinës.

## 5. Sfidat kryesore llogaritëse

- **Hapësira eksponenciale e kërkimit:** numri i ecjeve të mundshme rritet shpejt me \(N\).
- **Kufizimet e forta:** lëvizjet lokale dhe operatorët gjenetikë mund të shkaktojnë mbivendosje ose të prishin lidhshmërinë.
- **Minimat lokale:** për të fituar kontakte të reja mund të nevojitet humbja e përkohshme e disa kontakteve ekzistuese.
- **Ndjeshmëria ndaj parametrave:** orari i temperaturës, madhësia e popullatës dhe operatorët ndikojnë në cilësinë dhe kohën.
- **Projektimi i operatorëve:** mutacioni dhe kryqëzimi duhet të prodhojnë larmi pa krijuar shumë struktura të pavlefshme.
- **Krahasimi i drejtë:** algoritmet stokastike kërkojnë shumë prova dhe buxhete të krahasueshme llogaritëse.
- **Llogaritja efikase e energjisë:** pas një lëvizjeje lokale, ndryshojnë vetëm disa kontakte; rillogaritja e të gjitha çifteve mund të jetë e panevojshme.

## 6. Metodologjia e rekomanduar

1. **Përcaktoni përfaqësimin.** Shndërroni sekuencën në etiketat H/P dhe paraqitni konformacionin me koordinata ose me lëvizje në rrjetë. Fiksoni pikën dhe orientimin fillestar për të mos kërkuar përsëri struktura ekuivalente nga zhvendosja apo rrotullimi.
2. **Ndërtoni validuesin e strukturës.** Kontrolloni lidhshmërinë, vetëshmangien, koordinatat unike dhe përputhjen e pozicioneve me sekuencën. Testojeni veçmas me raste të vogla të njohura.
3. **Ndërtoni funksionin e energjisë.** Numëroni vetëm kontaktet H–H jo fqinje në sekuencë. Verifikojeni me struktura të vogla ku rezultati mund të llogaritet me dorë.
4. **Zvogëloni kërkimin e panevojshëm.** Përdorni thyerje simetrie, shmangni kthimet e menjëhershme kur është e përshtatshme dhe përdorni lëvizje lokale të vlefshme, si lëvizjet e skajeve, rrotullimet e qosheve ose lëvizjet crankshaft. Çdo reduktim duhet të ruajë objektivin dhe të mos përjashtojë pa arsye zgjidhje të vlefshme.
5. **Implementoni algoritmet me komponentë të përbashkët.**
   - **Simulated Annealing (SA):** gjeneron ndryshime lokale, pranon përmirësimet dhe pranon ndonjëherë zgjidhje më të këqija sipas një probabiliteti që varet nga temperatura.
   - **Genetic Algorithm (GA):** mban një popullatë, zgjedh prindër, aplikon mutacion dhe kryqëzim, pastaj ruan kandidatët e mirë. Pasardhësit duhet të jenë të vlefshëm ose të validohen dhe riparohen.
   - **GA hibrid me SA:** përdor SA për të përmirësuar lokalisht individë të zgjedhur, për shembull pas krijimit të pasardhësve ose para rikthimit të tyre në popullatë. Buxheti total i vlerësimeve duhet të jetë i krahasueshëm me algoritmet e tjera.
6. **Analizoni kompleksitetin.** Për \(N\) mbetje, llogaritja e plotë me kontroll të të gjitha çifteve kushton \(O(N^2)\). Përdorimi i një harte koordinatë-rrjetë ose numërimi përmes fqinjëve mund ta ulë një vlerësim të plotë në \(O(N)\). Ruajtja e një strukture kërkon \(O(N)\) hapësirë; popullata GA me madhësi \(P\) kërkon \(O(PN)\). Për \(K\) propozime SA, vlerësimi i plotë jep kufi \(O(KN^2)\), ndërsa përditësimet lokale të energjisë mund ta ulin koston për propozim. GA me \(G\) gjenerata dhe popullatë \(P\) ka rreth \(O(GP)\) vlerësime fitness-i, pa përfshirë kostot e operatorëve dhe kërkimit lokal.
7. **Krahasoni eksperimentalisht.** Ekzekutoni çdo algoritëm disa herë për të njëjtat sekuenca. Raportoni energjinë më të mirë, mesatare dhe shpërndarjen e saj, kohën, numrin e vlerësimeve dhe ecurinë e konvergjencës. Dokumentoni parametrat dhe farat e rastësisë. Përdorni buxhete të barabarta vlerësimesh ose kohe dhe shqyrtoni ndikimin e gjatësisë së sekuencës.

## 7. Hyrjet dhe daljet e pritshme

### Hyrjet

- Sekuenca e aminoacideve ose, për bërthamën e modelit, sekuenca H/P.
- Harta e dokumentuar nga aminoacidet në H/P, nëse hyrja përdor alfabetin e plotë të aminoacideve.
- Algoritmi dhe parametrat e tij, si kufiri i iteracioneve/gjeneratave, madhësia e popullatës dhe orari i temperaturës.
- Fara opsionale e rastësisë dhe parametrat e eksperimentit.

### Daljet

- Konformacioni më i mirë i vlefshëm, si koordinata ose si lëvizje në rrjetë.
- Sekuenca H/P, energjia dhe numri i kontakteve H–H.
- Rezultatet e validimit për lidhshmërinë dhe vetëshmangien.
- Parametrat e ekzekutimit, numri i vlerësimeve, koha dhe të dhënat e konvergjencës.
- Për eksperimentet, rezultatet e agreguara dhe krahasimi i përsëritjeve.

Pra, rezultati është një konformacion i optimizuar në rrjetën 2D sipas modelit HP, së bashku me dëshmi se ai plotëson kufizimet strukturore të modelit.
