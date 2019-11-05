# FindBestFood
Planing to go out for dinner tonight? Tired of checking the reviews to see which one you would love? Here is the application that would do it for you!!! 

## Objective:
Sometimes it was hard to decide which hotel to go for dinner. I rather wanted to spend my time developing something to automize it.
This application compares the best in the 2 hotels specified. Percentage of positive VS negative reviews, Most frequent words that might be useful to infer and top reviews which says most of what we have to know in small sentences.

## Steps:

1) Get the information from the UI with '&' to seperate 2 restaurant and 'in' to specify the location.
2) By parsing through '&', 'in' it will identify what we are looking for. [Next step - if, there is no & present find the objective for one hotel]
3) Then parse in the website to read the reviews of each website.
4) Cleaning using NLP - NER, stopwords removal, removing unrelated words and symbols.
5) Finding the most common words based on POS(Part of speach) was extremly helpful for finding the most related words.
6) Based on top frequent words and finding which sentence has more frequent item also not a largest among all the sentences found are considered for top reviews.


## Preview Of FindBestFood
![](HotelFinder.gif)
