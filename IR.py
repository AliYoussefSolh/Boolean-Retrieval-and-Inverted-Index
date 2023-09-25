# -*- coding: utf-8 -*-
"""
Created on Sat Sep 16 14:19:48 2023

@author: HES
"""

# =============================================================================
# import nltk
# =============================================================================
import os
import glob
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from bs4 import BeautifulSoup
import matplotlib.pyplot as plt
import sys
import time
import re

# =============================================================================
# nltk.download('punkt')
# nltk.download('stopwords')
# =============================================================================
from collections import defaultdict


def read_inverted_index_from_file(filename):
    inverted_index = defaultdict(lambda: {'doc_freq': 0, 'postings': []})

    with open(filename, 'r', encoding='utf-8') as file:
        for line in file:
            first_word_match = re.match(r'(\w+)', line)
          
            term = first_word_match.group(1)
            
            number_match = re.search(r'\b(\d+)\b', line)
           
            doc_freq = int(number_match.group(1))
            
            list_match = re.search(r'\[([\d,\s]+)\]', line)
            list_string = list_match.group(0).strip('[]').split(', ')
            postings=[]
            for integer in list_string:
                postings.append(int(integer))
            
            
            # parts = line.strip().split(' ')
            # term = parts[0]
            # doc_freq = int(parts[1])
            # print(parts[2])
            # # Properly parse the postings list
            # postings = [int(posting) for posting in parts[2].strip('[]').split(', ') if posting]

            inverted_index[term] = {
                'doc_freq': doc_freq,
                'postings': postings
            }

    return inverted_index;



def save_mappings_from_documents_ids_to_filename(document_id_to_filename):
    with open('document_id_to_filename_mapping.txt', 'w') as file:
        for doc_id, filename in document_id_to_filename.items():
            line = f"{doc_id} {filename}\n"
            file.write(line)
            
def save_inverted_index_to_file(inverted_index, filename):
    with open(filename, 'w',encoding='utf-8') as file:
        for term, info in inverted_index.items():
            line = f"{term} {info['doc_freq']} {info['postings']}\n"
            file.write(line)
            
            
            
def preprocess(text):
    # Tokenize the text into words
    words = word_tokenize(text.lower())
    
    # Remove stopwords and non-alphabetic words
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.isalpha() and word not in stop_words]
    # Stemming
    ps = PorterStemmer()
    words = [ps.stem(word) for word in words]
    
    return words            
                      
def read_html_documents(directory):
    def extract_text_from_html(html_file):
        try:
            with open(html_file, 'r', encoding='utf-8') as file:
                soup = BeautifulSoup(file, 'html.parser')
                return soup.get_text()
        except UnicodeDecodeError:
            print(f"Could not decode {html_file} with UTF-8. Skipping.")
            return None
    documents = {}
    document_id_to_filename = {}
    idx=0
    html_collection_size=0
    counter=0
    html_files = glob.glob(os.path.join(directory, '**/*.html'), recursive=True)
    for filepath in html_files:
        counter+=1
        print(counter)
        file = os.path.basename(filepath)
        document_id_to_filename[idx] = file
        document_distinct_and_stemmed_words= extract_text_from_html(filepath)
        if document_distinct_and_stemmed_words is None:
            continue
        documents[idx]=preprocess(document_distinct_and_stemmed_words)
        idx=idx+1
        html_collection_size+=os.path.getsize(filepath)
      
            
    return documents,document_id_to_filename,html_collection_size


def interate_over_inverted_index(inverted_index):
    for term, data in inverted_index.items():
        print("Term:", term)
        print("Document Frequency:", data['doc_freq'])
        print("Postings:", data['postings'])
    

def build_inverted_index(documents):
    inverted_index = defaultdict(lambda: {'doc_freq': 0, 'postings': []})
    for doc_id, terms in documents.items():
        for term in set(terms):  # Count each term only once per document
            inverted_index[term]['doc_freq'] += 1
            inverted_index[term]['postings'].append(doc_id)
    
    return inverted_index





def calculate_statistics(inverted_index,documents):
    num_distinct_words = len(inverted_index)
    num_documents = len(documents)
    total_distinct_words = sum(len(doc) for doc in documents.values())
    average_words_per_document = total_distinct_words / num_documents
    index_size_bytes = sys.getsizeof(inverted_index)
    #len(str(inverted_index).encode('utf-8'))
    document_frequencies = [info['doc_freq'] for info in inverted_index.values()]
    #additional
    average_document_frequency = sum(document_frequencies) / len(document_frequencies)
    most_frequent_word = max(inverted_index, key=lambda term: inverted_index[term]['doc_freq'])
    return num_distinct_words, num_documents, average_words_per_document,index_size_bytes,average_document_frequency,most_frequent_word

#histogram data
def doc_frequencies(inverted_index):
    document_frequencies = [info['doc_freq'] for info in inverted_index.values()]
    data_freq={}
    for freq in document_frequencies:
        if freq in data_freq:
            data_freq[freq]=data_freq[freq]+1
        else:
            data_freq[freq]=1
        
    return data_freq  
        

def histogram(inverted_index):
    data = doc_frequencies(inverted_index)
    x=list(data.keys ())
    y=list(data.values())
    #doc_frequencies= [info['doc_freq'] for info in inverted_index.values()]
    plt.bar(x,y, width=1.0, align='center')
    #plt.hist(doc_frequencies, bins=range(1, max(doc_frequencies) + 1), edgecolor='black')
    plt.xlabel('Document Frequency')
    plt.ylabel('Number of Terms')
    plt.title('Histogram of Document Frequencies')
    x_range = range(1, max(x) + 1, 300)
    plt.xticks(x_range)
    plt.yscale('log')
    
    # Adjust the x-axis and y-axis limits to increase the scale
    #plt.xlim(0, max(doc_frequencies) + 1)  # Set x-axis limits
    
    # plt.show()
    plt.savefig('histogram.pdf',dpi=300)
    #plt.saveFigure
    
    
# produce 26 lists: the list of all documents containing
#a word starting with a, the list of all documents containing a word starting with
#b, and so on until z.  
def generate_vertex_index(inverted_index):
    vertex_index = {chr(i): [] for i in range(ord('a'), ord('z') + 1)}

    for term, info in inverted_index.items():
        first_letter = term[0]
        if first_letter in vertex_index:
            vertex_index[first_letter].extend(info['postings'])

    # Deduplicate document IDs for each letter
    for letter in vertex_index:
        vertex_index[letter] = list(set(vertex_index[letter]))

    return vertex_index 

def save_vertex_index_to_file(vertex_index, filename):
    with open(filename, 'w') as file:
        for letter, doc_ids in vertex_index.items():
            file.write(f"{letter}\t{' '.join(map(str, doc_ids))}\n")

#inteserction algorithm in the slides 
#O(nlogn)           
def intersection_slides(list1, list2):
    # Sort the input lists
    list1.sort()
    list2.sort()
    
    # Initialize indices and the result set
    i, j = 0, 0
    result = []
    
    # Iterate through the sorted lists to find common elements
    while i < len(list1) and j < len(list2):
        if list1[i] == list2[j]:
            result.append(list1[i])
            i += 1
            j += 1
        elif list1[i] < list2[j]:
            i += 1
        else:
            j += 1
    
    return result

#O(n)
# The time complexity of the above program is O(n), where n is the length of the longer list between lst1 and lst2.

# The space complexity of the program is O(n), where n is the length of the smaller list between lst1 and lst2. 
def efficient_list_intersection(list1, list2):
    return list(set(list1) & set(list2))

def intersection_comparisons(vertex_index):
    # Generate pairs of lists from the vertex index (26 lists)
    pairs_of_lists = []

    for i in range(ord('a'), ord('z') + 1):
        for j in range(i + 1, ord('z') + 1):
            list1 = vertex_index[chr(i)]
            list2 = vertex_index[chr(j)]
            pairs_of_lists.append((list1, list2))
    # Calculate intersection for each pair using both algorithms and measure time
    # print(len(pairs_of_lists))
    total_elements = 0
    total_time_inefficient = 0
    total_time_efficient = 0

    for list1, list2 in pairs_of_lists:
        total_elements += len(list1) + len(list2)

        # Measure time for the inefficient algorithm
        start_time = time.time()
        result_inefficient = intersection_slides(list1, list2)
        total_time_inefficient += time.time() - start_time

        # Measure time for the efficient algorithm
        start_time = time.time()
        result_efficient = efficient_list_intersection(list1, list2)
        total_time_efficient += time.time() - start_time

         #Check if both algorithms give the same result
        # print(result_efficient)
        # print(result_inefficient)
        if set(result_inefficient) != set(result_efficient):
             print("Algorithms did not give the same result for pair")
             
    rate_inefficient = total_elements / (total_time_inefficient) 
    rate_efficient = total_elements / (total_time_efficient)

        # Print the rates
    print("Inefficient Algorithm Rate (elements per second):", rate_inefficient)
    print("Efficient Algorithm Rate (elements per second):", rate_efficient)

    
def reading_index_vertex_from_file_stats():
        # Assuming the vertex index is stored in a file
    vertex_index_file = "vertix_index.txt"
    
    # Measure the time to read the vertex index from disk
    start_time = time.time()
    
    # Read the vertex index from the file
    with open(vertex_index_file, 'r') as file:
        vertex_index_contents = file.readlines()
    
    # Calculate the time to read the vertex index in seconds
    total_time_read_vertex_index = time.time() - start_time
    
    # Calculate the rate (elements per second) for reading the vertex index
    # Assuming each line represents a list of document IDs
    total_elements_vertex_index = sum(len(line.split()[1:]) for line in vertex_index_contents)
    rate_read_vertex_index = total_elements_vertex_index / (total_time_read_vertex_index)
    # Print the rates for reading the vertex index from disk
    print("Rate for Reading Vertex Index from Disk (elements per second):", rate_read_vertex_index)
    print("Programming Language: Python")
    print("Machine Specifications: [Include relevant specifications here]")
    
def boolean_query_intersection(query, inverted_index):
    # Split the query into terms and operators
    query_terms = query.split()
    operators = []
    terms = []

    # Separate terms and operators
    for term in query_terms:
        if term in ['AND', 'OR', 'NOT']:
            operators.append(term)
        else:
            terms.append(term)

    # Initialize the result with the postings of the first term
    result = inverted_index.get(terms[0], {})['postings']

    i = 0  # Start from the first operator
    while i < len(operators):
        if operators[i] == 'AND':
            result = efficient_list_intersection(result, inverted_index.get(terms[i + 1], {})['postings'])
        elif operators[i] == 'OR':
            result = result + inverted_index.get(terms[i + 1], {})['postings']
        elif operators[i] == 'NOT':
            not_postings = inverted_index.get(terms[i + 1], {})['postings']
            result = [doc_id for doc_id in result if doc_id not in not_postings]

        i += 1

    return list(set(result))


def process_queries(inverted_index):
    query_runtimes = []
    sample_queries = [
    'bachelor AND learn AND link AND certificateearn',
    'googl OR coursera AND best',
    'london AND abroad AND internship NOT slow',
    'master OR coach OR strength OR influenc',
    'spread OR robust AND qualiti NOT stress',
    'studi AND facebook AND group OR member']

    for query in sample_queries:
        start_time = time.time()  # Record start time

    # Process the query and retrieve the documents satisfying the query
        result = boolean_query_intersection(query, inverted_index)

        end_time = time.time()  # Record end time

    # Calculate the runtime for this query
        query_runtime = end_time - start_time
        query_runtimes.append(query_runtime)

    # Print the query and the documents satisfying the query
        print(f"Query: {query} => Documents: {result}")

# Calculate the average runtime
    average_runtime = sum(query_runtimes) / len(query_runtimes)
    print(f"Average Runtime Of queries: {average_runtime} seconds")



    
    

    


def main():
    inv=read_inverted_index_from_file("index.txt")
    #print(inv)
    #print("[1, 2, 3]".strip('[]').split(", "))
    histogram(inv)
    
    # corpus_dir = "C:\\Users\\HES\\Desktop\\Corpus\\IR_Assignment"
    # #"C:\\Users\\HES\\Desktop\\test_IR"
    # documents,id_document_mapping,html_collection_size = read_html_documents(corpus_dir)
    # print(len(documents))
    # save_mappings_from_documents_ids_to_filename(id_document_mapping)
    # index= build_inverted_index(documents)
    # print("HTML CORPUS SIZE (BYTES): ",html_collection_size)
    # print("####################################################")
    # num_distinct_words, num_documents, average_words_per_document,index_size_bytes,average_document_frequency,most_frequent_word= calculate_statistics(index,documents)
    # print ("Number Of Distinct Words: ",num_distinct_words)
    # print("Number Of Documents: ",num_documents)
    # print ("Average Words Per Documnet: ",average_words_per_document)
    # print("Index Size (Bytes): ",index_size_bytes)
    # print("Average Document Frequency : ",average_document_frequency)
    # print("Most Frequent Word : ",most_frequent_word)
    # histogram(index)
    # print("####################################################")
    # save_inverted_index_to_file(index, "index.txt")
    # vertex_index = generate_vertex_index(index)
    # save_vertex_index_to_file(vertex_index, "vertix_index.txt")
    # intersection_comparisons(vertex_index)
    # print("####################################################")
    # reading_index_vertex_from_file_stats()
    # #print(index.get("linkedinhow",{}))
    # #print(boolean_query_intersection("within OR practition NOT exactli",index))
    # process_queries(index)
    
    
main()

    


