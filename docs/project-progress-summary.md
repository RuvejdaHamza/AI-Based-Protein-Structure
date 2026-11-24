# Përmbledhje e punës së projektit

## AI-Based Protein Structure Prediction and Optimization

Ky dokument përmbledh dizajnin dhe implementimin e projektit të optimizimit të palosjes së proteinave me modelin 2D Hydrophobic–Polar (HP), përfshirë përpunimin e sekuencës, validimin, energjinë, algoritmet SA/GA/hibrid, reduktimin e hapësirës së kërkimit, testet dhe matjet e performancës.

## 1. Analiza algoritmike

Problemi është formuluar si optimizim kombinator mbi konformacione të zinxhirit në një rrjetë katrore 2D. Çdo aminoacid klasifikohet si hidrofobik \(H\) ose polar \(P\). Konformacioni duhet të ruajë lidhshmërinë e zinxhirit dhe vetëshmangien. Funksioni bazë i energjisë favorizon kontaktet H–H mes mbetjeve jo të njëpasnjëshme:

\[
E(X)=-\epsilon C_{HH}(X),
\]

ku \(C_{HH}\) është numri i kontakteve dhe \(\epsilon>0\) është forca e ndërveprimit. Janë diskutuar përshtatshmëria dhe kufizimet e modelit, sfidat e kërkimit, kompleksiteti, metodologjia eksperimentale dhe hyrjet/daljet e pritshme.

Dokumenti: [algorithmic-analysis.md](./algorithmic-analysis.md)

## 2. Arkitektura e softuerit

Është përshkruar arkitektura modulare për implementim në një Kaggle Notebook. Ajo përfshin konfigurimin, përpunimin e sekuencës, përfaqësimin HP, gjenerimin dhe validimin e strukturave, energjinë, operatorët lokalë, SA, GA, metodën hibride, vizualizimin, matjen e performancës dhe krahasimin eksperimental.

Arkitektura përcakton kontrata të përbashkëta për sekuencën, konformacionin, rezultatet e optimizimit dhe eksperimentet. Ajo rekomandon të ndahen algoritmet nga vizualizimi dhe raportimi, të ruhen parametrat/farat e rastësisë dhe të krahasohen algoritmet me buxhete llogaritëse të krahasueshme.

Dokumenti: [software-architecture.md](./software-architecture.md)

## 3. Modeli 2D HP

Është përcaktuar modeli i rrjetës katrore dhe klasifikimi i aminoacideve:

- **Hidrofobike (H):** A, C, G, I, L, M, F, P, W, Y, V.
- **Polare (P):** R, N, D, Q, E, K, S, T, H.

Konvertimi i sekuencës ruan rendin dhe gjatësinë. Çdo hap në rrjetë përdor një nga drejtimet relative \(R,L,U,D\). Koordinatat gjenerohen në mënyrë kumulative. Janë formalizuar vetëshmangia, lidhshmëria e zinxhirit, kontaktet jo të njëpasnjëshme H–H dhe simetria e rrotullimit/pasqyrimit.

Dokumenti: [hp-lattice-model.md](./hp-lattice-model.md)

## 4. Moduli i përpunimit të sekuencës

Moduli [sequence_processing.py](../src/hp_folding/sequence_processing.py) ofron:

- normalizimin e shkronjave dhe heqjen e hapësirave;
- validimin e 20 kodeve standarde të aminoacideve;
- konvertimin në sekuencë H/P sipas hartës së dokumentuar;
- llogaritjen automatike të gjatësisë;
- mesazhe gabimi për hyrje boshe, simbole të pavlefshme dhe tipe të gabuara.

Rezultati përfaqësohet nga `ProcessedSequence`, me sekuencën e normalizuar, sekuencën HP dhe gjatësinë. Testet janë në [test_sequence_processing.py](../tests/test_sequence_processing.py).

## 5. Përfaqësimi dhe validimi i strukturës

Moduli [structure.py](../src/hp_folding/structure.py) përfaqëson proteinën si koordinata të renditura \((x,y)\), një koordinatë për çdo mbetje. Mbështet ndërtimin e koordinatave nga drejtimet relative, paraqitje tekstuale dhe hartëzim koordinatë–pozicion mbetjeje.

Validuesi kontrollon gjatësinë e pritshme, koordinatat, lidhshmërinë Manhattan prej një njësie dhe mungesën e mbivendosjeve. Gjeneruesi krijon ecje vetëshmangëse të rastësishme me kufij të qartë kërkimi; nuk kthen strukturë të pavlefshme si zëvendësim të heshtur. `ValidationStatistics` ruan numrin e strukturave të kontrolluara, të vlefshme/të pavlefshme dhe kategoritë e gabimeve. Testet janë në [test_structure.py](../tests/test_structure.py).

## 6. Vlerësimi i energjisë

Moduli [energy.py](../src/hp_folding/energy.py) implementon

\[
E=-\epsilon C_{HH}.
\]

Për çdo mbetje H kontrollohen fqinjët e saj në rrjetë. Kontakti numërohet vetëm nëse mbetja tjetër është H dhe nuk është fqinje në sekuencë; indeksi i rendit përdoret për të shmangur numërimin e dyfishtë. Llogaritja është \(O(N)\) për \(N\) mbetje me hartën koordinatë–indeks. Funksioni refuzon sekuenca HP ose struktura të pavlefshme dhe kthen `EnergyResult` me energjinë dhe numrin e kontakteve. Testet me shembuj të verifikueshëm me dorë janë në [test_energy.py](../tests/test_energy.py).

## 7. Operatorët lokalë të modifikimit

Moduli [local_moves.py](../src/hp_folding/local_moves.py) përmban:

- **Lëvizjen e skajit:** zhvendos një mbetje terminale në një pikë të lirë fqinje me ankorën e saj.
- **Lëvizjen e qoshes:** rrotullon një mbetje të brendshme në kulmin alternativ të katrorit, duke ruajtur lidhjet.
- **Lëvizjen crankshaft:** rikonfiguron dy mbetje të brendshme mes ankorave fikse.
- **Gjenerimin e fqinjësisë:** mbledh kandidatët unikë të vlefshëm të arritshëm me një lëvizje.

Çdo propozim kontrollohet për përplasje dhe validohet. Strukturat burimore të pavlefshme refuzohen me gabim, pa riparim automatik. `MoveStatistics` mat lëvizjet e tentuara, të pranuara dhe të refuzuara. Testet janë në [test_local_moves.py](../tests/test_local_moves.py).

## 8. Gjendja aktuale dhe hapat e ardhshëm

Të implementuara deri tani janë përpunimi i sekuencës, përfaqësimi/validimi i strukturës, gjenerimi i strukturave, energjia, operatorët lokalë dhe Simulated Annealing, bashkë me testet përkatëse.

### Simulated Annealing

Moduli [simulated_annealing.py](../src/hp_folding/simulated_annealing.py) nis nga një strukturë e vlefshme, llogarit energjinë fillestare, gjeneron fqinjësinë me operatorët lokalë, zgjedh një kandidat të vlefshëm dhe zbaton rregullin Metropolis. Përdor ftohje gjeometrike \(T_i=T_0\alpha^i\), gjurmon strukturën/energjinë më të mirë, historikun, kohën dhe numëruesit e lëvizjeve/kandidateve.

Rezultati `SimulatedAnnealingResult` përfshin iteracionet, kandidatët e vlefshëm/të pavlefshëm, lëvizjet e pranuara/refuzuara, vlerësimet e energjisë, energjitë fillestare/përfundimtare/më të mira, historikun dhe arsyen e ndalimit. Fqinjësia mund të kufizohet në lloje lëvizjesh të zgjedhura. Përdoret gjurmuesi inkremental i energjisë, me matje të kontakteve të kontrolluara dhe propozimeve të përsëritura të shmangura. Kushtet e ndalimit përfshijnë kufirin e iteracioneve, temperaturën minimale, objektivin e energjisë, mungesën e fqinjëve të vlefshëm dhe kufirin pa përmirësim. Testet janë në [test_simulated_annealing.py](../tests/test_simulated_annealing.py).

Për \(N\) mbetje, fqinjësia përmban \(O(N)\) propozime dhe verifikimi i secilit kushton \(O(N)\). Prandaj një iteracion kushton \(O(N^2)\), ndërsa \(I\) iteracione kushtojnë \(O(IN^2)\); ruajtja e fqinjësisë kërkon deri në \(O(N^2)\) hapësirë.

### Genetic Algorithm

Moduli [genetic_algorithm.py](../src/hp_folding/genetic_algorithm.py) kodon individët si \(N-1\) drejtime relative. Popullata fillestare krijohet me gjeneruesin e ecjeve vetëshmangëse. Përzgjedhja bëhet me tournament selection dhe energjia më e ulët fiton. Kryqëzimi me një pikë bashkon segmente të vargut të drejtimeve; mutacioni zëvendëson lëvizje të zgjedhura me një nga drejtimet e tjera të rrjetës.

Kryqëzimi dhe mutacioni nuk supozohen të prodhojnë struktura të vlefshme: pasardhësit dekodohen hap pas hapi, përplasjet zbulohen herët dhe kandidati validohet. Pasardhësit e pavlefshëm numërohen dhe refuzohen; pas numrit të caktuar të provave, algoritmi raporton gabim nëse nuk mund të plotësojë gjeneratën. Elitizmi kopjon individët më të mirë pa ndryshim. Dublikimet e gjenomit ose strukturës simetrikisht ekuivalente shmangen nga vlerësimi i përsëritur, kur aktivizohet `symmetry_reduction`. Rezultati përmban kandidatë të krijuar/të vlefshëm/të pavlefshëm, përplasje të refuzuara herët, hapa koordinatash të kursyer, dublikime, vlerësime energjie, energji më të mirë/mesatare, diversitet, kohë dhe histori për gjeneratë. Parametrat përfshijnë kufirin e gjeneratave, objektivin e energjisë dhe ndalimin pas mungesës së përmirësimit. Testet janë në [test_genetic_algorithm.py](../tests/test_genetic_algorithm.py).

Për popullatë \(P\), zinxhir me \(N\) mbetje dhe \(G\) gjenerata: vlerësimi i energjisë dhe validimi kushtojnë \(O(N)\) për individ, përzgjedhja tournament \(O(k)\) për prind me turne të madhësisë \(k\), dhe riprodhimi i një gjenerate kushton zakonisht \(O(PN)\), pa përfshirë riprovimet për pasardhës të pavlefshëm. Memoria për popullatën është \(O(PN)\); në rastin më të keq provat e kufizuara e shumëzojnë punën sipas kufirit të riprovimeve.

### Algoritmi hibrid GA + SA

Moduli [hybrid_optimizer.py](../src/hp_folding/hybrid_optimizer.py) ripërdor përzgjedhjen, kryqëzimin, mutacionin dhe vlerësimin e GA-së, si dhe implementimin e plotë të Simulated Annealing. Pas krijimit të çdo gjenerate, sipas frekuencës së konfiguruar përzgjidhet fraksioni më premtues i popullatës për rafinim lokal me SA. Një individ zëvendësohet vetëm kur SA gjen energji më të ulët. Elitizmi ruan kandidatin global më të mirë. Të gjitha strukturat mbeten të validuara.

Rezultati ndan kandidatët/vlerësimet e GA-së nga kandidatë dhe kosto të SA-së, dhe raporton koston totale, kohën e rafinimit lokal, numrin e thirrjeve/zëvendësimeve, energjitë fillestare e përfundimtare të popullatës, historikun para/pas rafinimit, diversitetin dhe përmirësimin ndaj popullatës fillestare. Frekuenca dhe fraksioni i individëve për SA janë të konfigurueshëm. Testet janë në [test_hybrid_optimizer.py](../tests/test_hybrid_optimizer.py).

Për popullatë \(P\), gjatë \(G\) gjeneratave, gjatësinë \(N\), numrin mesatar \(L\) të individëve të rafinuar për gjeneratë dhe \(I\) iteracione SA për individ: kostoja bazë e GA-së është zakonisht \(O(GPN)\), ndërsa SA shton deri në \(O(GLIN^2)\), duke supozuar rafinim në çdo gjeneratë; frekuenca më e ulët e rafinimit e zvogëlon këtë komponent proporcionalisht. Memoria është \(O(PN+N^2)\) në implementimin aktual, sepse SA materializon fqinjësinë.

#### Eksperiment krahasues demonstrues

U ekzekutuan pesë prova me të njëjtën sekuencë `HPPHHPHHPPHH` dhe farat `11, 23, 37, 41, 59`. GA dhe hibridi përdorën popullatë 16, 15 gjenerata, turne 3 dhe elitizëm 1. Hibridi rafinoi 25% të individëve çdo 3 gjenerata me SA lokale deri në 10 iteracione. SA e pavarur përdori ecje fillestare të rastësishme dhe deri në 120 iteracione. Kjo është një provë demonstrative e vogël, jo provë statistikore përfundimtare; buxhetet e llogaritjes ndryshojnë, ndaj koha dhe vlerësimet raportohen së bashku me energjinë.

| Algoritmi | Energjia mesatare më e mirë | Energjia më e mirë | Koha mesatare (s) | Vlerësime mesatare energjie |
|---|---:|---:|---:|---:|
| GA | -4.6 | -5 | 0.0256 | 142 |
| SA | -3.8 | -5 | 0.1584 | 121 |
| GA + SA hibrid | -4.4 | -5 | 0.2626 | 366.6 |

Në këto pesë prova, hibridi arriti energjinë më të mirë të vërejtur, por mesatarja e GA-së së pastër ishte pak më e ulët dhe hibridi kërkoi më shumë kohë/vlerësime. Kjo tregon se rafinimi lokal shton kosto dhe nuk garanton përmirësim ndaj GA-së; rezultatet varen nga parametrat dhe sekuenca. Matjet janë përsëritur pas shtimit të kontrollit të dublikimeve.

## Reduktimi i hapësirës së kërkimit

Dokumenti [search-space-reduction.md](./search-space-reduction.md) dhe moduli [search_space.py](../src/hp_folding/search_space.py) përshkruajnë/implementojnë simetrinë D4, identifikimin e gjendjeve dublikatë, refuzimin e hershëm të përplasjeve, përditësimet inkrementale të energjisë, fqinjësitë e kufizuara, elitizmin dhe ndalimin e hershëm. Ai dallon kandidatët e pavlefshëm, dublikatat e vlefshme që nuk rivlerësohen, mundësitë e operatorëve të lëna jashtë dhe kandidatët që vlerësohen. Numri i saktë i gjendjeve që nuk gjenerohen nuk pretendohet, sepse hapësira e plotë nuk eksplorohet.

Matjet kryesore në pesë prova të njëjta:

- GA zbuloi 566 kandidatë të pavlefshëm nga 807 të gjeneruar dhe shmangu mesatarisht 2,871 hapa drejtimi përmes zbulimit të hershëm të përplasjeve.
- Në këtë grup provash, kanonizimi i simetrisë nuk gjeti dublikata shtesë (0), nuk uli vlerësimet mesatare (142), dhe pati kosto kohore: 0.0257 s kundrejt 0.0130 s pa simetri. Testi i njësisë verifikon se një çift i rrotulluar/pasqyruar kapet si dublikatë.
- Kufizimi i SA nga `end+corner+crankshaft` te `end+corner` uli kohën mesatare nga 0.1624 s në 0.0913 s, por energjia më e mirë mesatare u përkeqësua nga -3.8 në -3.4.
- Për përditësimin inkremental, u numëruan mesatarisht 456 kontrolle fqinjësh kontakti kundrejt 3,388 kontrolleve të vlerësimit të plotë (vlerësim teorik; validimi mbetet kosto shtesë \(O(N)\)).

## Optimizimi dhe matja e performancës

Analiza e plotë asimptotike, me rastet më të mira/më të këqija, kufijtë e
provave të përsëritura dhe kostot e mekanizmave të reduktimit, gjendet te
[complexity-analysis.md](./complexity-analysis.md).

Protokolli për eksperimentet e ardhshme—me sekuenca sintetike dhe fragmente
reale, seed-e fikse, buxhete të përbashkëta, metrika dhe analizë statistikore—
gjendet te [experimental-evaluation-protocol.md](./experimental-evaluation-protocol.md).
Është plan eksperimental; nuk paraqet rezultate të paekzekutuara.

Runner-i eksperimental është implementuar te
[experiment_runner.py](../src/hp_folding/experiment_runner.py). Ai ndërton 40
rastet e protokollit, ekzekuton SA/GA/hibrid me 10 përsëritje dhe kufi të
përbashkët prej 200 vlerësimesh energjie, mat kohën/alokimin maksimal Python,
dhe eksporton rezultatet individuale e përmbledhëse në CSV. Vlerësimet e
energjisë janë buxheti i barabartë i kërkimit; koha reale mbetet e raportuar
sepse kostot e gjenerimit të kandidatëve ndryshojnë.

U optimizuan pikat e nxehta pa ndryshuar objektivin \(E=-\epsilon C_{HH}\):

- Validimi i strukturës kontrollon lidhshmërinë gjatë një kalimi të vetëm dhe shmang krijimin e listave të gabimeve për strukturat e vlefshme.
- Gjenerimi i fqinjësisë validon strukturën burimore një herë dhe përdor kontroll lokal të zënies së koordinatave për lëvizjet.
- Dekodimi i GA-së ndalon në përplasjen e parë; kandidati i vlefshëm përdor rrugën e brendshme të vlerësimit pa validim të përsëritur. API-ja publike vazhdon të validojë hyrjet.
- SA përdor gjurmuesin inkremental të kontakteve; zgjedhja e elitës përdor `heapq.nsmallest` në vend të renditjes së gjithë popullatës.
- Simetria D4 mbetet opsionale në GA, sepse çelësat kanonikë mund të kushtojnë më shumë se dublikatat që shmangin për popullata të vogla.

Benchmark-u i përsëritshëm është në [benchmark_performance.py](../benchmarks/benchmark_performance.py). Matjet u morën në mjedisin aktual; versionet bazë janë rrugë reference të rindërtuara, jo versione historike nga Git. Koha është mediana e pesë ekzekutimeve për testet e komponentëve; `tracemalloc` raporton kulmin e alokimeve Python, jo RSS të procesit.

| Testi | Rruga reference | Rruga e optimizuar | Vëzhgimi |
|---|---:|---:|---|
| Vlerësimi i energjisë, 48 mbetje | 0.000080 s; 3,737 B | 0.000017 s; 3,376 B | E njëjta energji; varianti i optimizuar merr strukturë të validuar nga thirrësi. |
| Gjenerimi i fqinjësisë | 0.010617 s; 29,664 B | 0.004161 s; 29,088 B | Të dyja prodhuan të njëjtat 20 struktura kanonike. |
| Vlerësimi GA, 250 kandidatë | 0.032100 s; 953,041 B | 0.007185 s; 953,016 B | 250 të vlefshëm, 0 të pavlefshëm dhe 250 vlerësime energjie në të dyja rrugët; e njëjta energji më e mirë. |
| SA, pesë fara | 0.045399 s mesatare | 0.021842 s mesatare | Energjia më e mirë mesatare -1.8, më e mira -3 dhe 36 vlerësime mesatare në të dyja. |

Këto rezultate sugjerojnë ulje kohe në këto prova, veçanërisht nga shmangia e validimit të përsëritur; nuk garantojnë përshpejtim në çdo madhësi apo mjedis. Ulja e alokimeve të pikut ishte e vogël në testet e GA-së, prandaj përfitimi kryesor i matur është koha. Detajet dhe kufizimet janë në [search-space-reduction.md](./search-space-reduction.md).

**Nuk janë implementuar ende:** vizualizimi i rezultateve dhe moduli i përgjithshëm i matjes/ruajtjes së eksperimenteve.

Hapat e sugjeruar:

1. Shtimi i vizualizimit të konformacioneve dhe historikut të konvergjencës.
2. Shtimi i modulit të përgjithshëm për ekzekutim, ruajtje dhe përmbledhje eksperimentesh.
3. Zgjerimi i provave me më shumë sekuenca, fara dhe buxhete të barabarta.
4. Krahasimi i cilësisë së energjisë, kohës, vlerësimeve dhe ndryshueshmërisë.

## 9. Verifikimi

Pas optimizimeve, u ekzekutua suita e plotë e testeve:

```text
Ran 86 tests
OK
```

Komanda për ta përsëritur nga direktoria e projektit:

```text
$env:PYTHONPATH = "src"
python -m unittest discover -s tests -v
```
