# Feedback Intermediate Presentation 08.12.2022

- TCP-H Datensatz benutzen zum Benchmarking
- Viel viel mehr Datensaetze
  - Setup zum Verarbeiten mehrerer Datensaetze, statt immer in /src/ folder zu kopieren -> Run-Folder/Output-Folder Ansatz
- IMDB Datensatz, BIOSQLSP, TCP-S

- Budget an unique Values, dass wir uns anschauen koennen (z.B. Anzahl Tuple, Arbeitsspeicher)
- Moeglichst gut ausnutzen mit verschiedenen Sampling Strategien
- Spaltenweise 'Informationsgehalt/Informationsdichte' bestimmen und anhand dessen Sampling Rate und Strategie bestimmen
- Werte mit Statistiken moeglichst gut aufraeumen
- FNs mit partialSPIDER reduzieren
- FPs mit Auswertung der MissingValues in Kombination mit Statistischen Kennzahlen reduzieren

- Unary INDs reichen aus, wenn wir das gut machen.
- Nary nur als Bonus am Ende