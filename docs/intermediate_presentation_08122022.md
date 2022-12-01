# Notes about intermediate presentation 08.12.2022

Tooling:

- 2 Haupt-Methoden:
  - sampling
    - Steuerung ueber CMD Args bzw. configuration.py Datei
    - Durchfuehrung Sampling
      - Fuer jede Sampling Methode:
        - Fuer jede Sampling Rate:
          - Fuer jede Eingabedatei:
            - Sampling durchfuehren, tmp-Datei abspeichern
    - Ausfuerung metanome mit Vielzahl von Konfigurationen
      - Bisher:
        - Je eine Version (entweder original oder eine der sampled) jeder Eingabedatei in Metanome geben und Ergebnisse parsen
    - Parsen des metanome outputs in selbst-definiertes JSON Format
      - Verbindet Output mit Run-Configuration
  - evaluation
    - Parsen JSON Output in CSV Datei
      - TP, FP, FN
      - Precision, Recall, F1-Score (Notiz: Accuracy kann nicht berechnet werden, da wir TN nicht berechnen wollen)
      - Fuer unary einzelner Wert
      - Fuer nary Liste mit einzelnen Werten pro arity
    - Erstellen von Plots
      - unary:
        - Stacked Barplot (TP, FP, FN)
          - detailed:
            - 1x gruppiert nach Sampling Methode (Welche Datei wurde mit welchem Verfahren gesampled)
            - 1x gruppiert nach Sampling Rate (Welche Datei wurde mit welcher Rate gesampled)
          - simple:
            - Gruppiert nach Anzahl der Dateien, die gesampled wurden
        - Line-Plot (Precision, Recall, F1-Score)
          - detailed & simple wie oben
      - nary:
        - Onion-Plot
          - Gruppiert nach Arity der detektierten INDs
          - Gruppiert nach Anzahl sampled Dateien
          - Ausgabe der TP
    - Erstellen von JSON mit tuples_to_remove Metrik
      - Bisher noch kein Parsing/Auswertung davon abgesehen von Konsolenausgabe

Sampling Methoden:

- Random
- First (erste x Zeilen aus der Datei)
- evenly-spaced (gleiche Abstaende zwischen ausgewaehlten Zeilen)

Visualisierungen:

- Visualisierungen vorher Umbauen! Dafuer muss Column Sampling gemergt sein.

Evaluationen:

- Datensaetze:
  - SCOP
  - CATH
