# Regression fixtures

PDF regression tests read selected pages of the supplied `data/raw/cdc.pdf`
directly: 5 (short article), 13 (article 30 and footnote 10), 30–31 (decimal
identifiers and quoted articles in notes), 52 (amendment notes), 161 (article
503 with an unstyled reference), 177 (Livre V), and 242–243 (articles 748–750).
The complete frozen corpus is also checked for all 839 article identifiers,
including 42-1 and the `bis` articles.
Small text fixtures in the tests cover generic layouts without fabricating PDF
provenance. No remote documents or credentials are needed.
