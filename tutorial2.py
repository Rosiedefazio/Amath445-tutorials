# Array in numpy is much much simpiler than a list in python when it comes the data storage 
#dont worrky about the 3, in numpy arry 
# can change the numpy elements by indexing
# what about creating a matrix, numpy is row major, unlike matlab and r which is coloumn major 
# can create three dimensioinall arrays as well (up to four dimension is okay, try to avoid for loops when using array) 
#for the first assignemnt two dimension is all you need 
# caution, try to make your code work in the most simpliest way possible and then if possible, optimize run time 
# will see 0. instead of 0 because by default numpy float64 
# when you start using pytorch, when you are training your data, float 64 can be too much memory for your computer? 
#numpy has concat which is a generalisation of row and column stack 
# indexing in mupy is different than python lists, data[nth row, nth column]
#because numpy is row major, first value is ro, and when you splice, the index for last number is not included ie data[0:2, 0]
#give rows 0 and 1 becasue 2 isnt included 
# in numpy you do not create a new copy of the data set if you adjust the data. Like SAS 
#bool index looks good for decision trees 
