## Co mamy
- Aplikacja webowa – biblioteka streamlit – 10 pkt
- Wielkość i przechowywanie analizowanego zbioru danych – 1.000 – 9.999 – 5 pkt - wyczyściłam duplikaty, więc teraz mamy mniej
- Przechowywanie wykorzystywanego zbioru danych w relacyjnych bazach danych (SQL Lite) - 10 pkt
- Dodatkowe unikatowe cechy w zbiorze danych – 6 dodatkowe cechy (razem mamy 16) – 3 pkt - po dodaniu kolumn mamy 16
- Możliwość wykorzystania filtra - według roku, gatunku, języka, czasu trwania, średnie oceny powyżej jakiegoś poziomu, przychody powyżej jakiegoś progu, ton – 7 pkt
- generowanie opisu według reguł - 10 pkt
- nadawanie wag termom - TF-IDF - 10 pkt
- miary podobieństwa - cosinusa, LSI, Jaccarda - 20 pkt
- Podobieństwo termów – do poprawy błędów – odległość Levensteina – 5 pkt
- Klasyfikacja dokumentów - KNN - 10 pkt
- Analiza tekstu - Analiza sentymentu - 10 pkt
- 2 wykresy - 8 pkt
- **dodałam SVM - jak twoim zdaniem będzie to spoko to mamy 118 punktów**

**Razem: 108**

tam w dodatkowych cechach miałyśmy 6 pkt, a za każdą dodatkową jest 0.5 pkt, czyli mamy 3 pkt, ja dodałam relacyjną bazę danych


## Kryteria 
( tutaj możemy odznaczać co mamy zrobione, też jak wpadną jakieś nowe pomysły, na razie po prostu przekleiłam to co mam w pliku)
- Aplikacja webowa – biblioteka streamlit – 10 pkt
- Wielkość i przechowywanie analizowanego zbioru danych – 10.000-25.000 (10.000) – 8 pkt
- Dodatkowe unikatowe cechy w zbiorze danych – 2 dodatkowe cechy (razem mamy 12) – 2 pkt 
- Możliwość wykorzystania filtra - według roku, gatunku, oceny, języka, czasu trwania, popularność powyżej jakiegoś poziomu, średnie oceny powyżej jakiegoś poziomu, przychody powyżej jakiegoś progu – 8 pkt
-  Generowanie/opisywanie wyników w postaci unikatowego tekstu – [opis rekomendowanych filmów - generowanie za pomocą API chatu gpt (na przykład znaleziono nam 10 filmów i na początku jest krótkie podsumowanie, że “To są najlepsze filmy świateczne i coś tam” (nie wiem czy rozumiem co masz na myśli – ja myślałabym bardziej tak ->)] – na podstawie informacji w kolumnach tworzymy gotowy opis z brakami, które ciągną się z tych kolumn – 10 pkt
-  Nadawnie wag termom - wyszukiwarka filmów na podstawie słów kluczowych - na przykład szukamy “świąteczny film” i na podstawie overview filmów znajdujemy te najbardziej pasujące (można też posortować wedlug rankingu) - coś podobnego co mieliśmy w zadaniu 5 - TF-IDF/ wykorzystanie osadzeń (embeddings) – 20 pkt
- Podobieństwo termów – do poprawy błędów – odległość Levensteina – 5 pkt
- Relewancja i miary efektywności (wyszukiwarki) – użycie np. miary prezycji – 10 pkt
- Miary podobieństw - miara cosinusa, LSI – 15 pkt
- Obliczanie ważności dokumentów – PageRank - zastanawiam się tylko czy nie miałoby to praktycznie takiej samej funkcjonalności co np. TF-IDF – więc można się jeszcze nad tym zastanowić – 10 pkt 
- Klasyfikacja dokumentów - Na podstawie ocen filmów tworzymy nową kolumnę dobry/zły (1 - dobry, jeżeli ocena jest powyżej 7, 0 - zły - jeżeli ocena jest poniżej 7). Wpisujemy informację o jakimś filmie którego nie ma w zbiorze danych (jego nazwę, overview) i system nam ocenia, czy ten film będzie dobry czy zły - możemy wziąć KNN, las losowy i naiwny klasyfikator Bayesa - 30 pkt 
- Analiza tekstu - Analiza sentymentu - stworzyć nową kolumnę, która będzie pokazywać, czy film ma pozytywny, negatywny czy neutralny ton – 10 pkt
Wychodzi razem 143 punkty, więc po pierwsze jest jeszcze miejsce na to, żeby coś wyrzucić, dwa ostatni punkt to wykresy jeśli wyrzucimy więcej i nam zabraknie można coś z tym wymyślić.

---------------------------------------------------------------------------------------------------------------
Na mojej gałęzi dodałam plik py, trochę tam zaczęłam coś robić: dodałam filtry i wyszukiwarkę z tf-idf.
Dodałam też plik csv z rekordami, jest ich więcej niż 10 000 tys, ale trzeba je jeszcze oczyścić, bo nie wiem czy nie ma tam jakiś pustych wierszy i duplikatów
