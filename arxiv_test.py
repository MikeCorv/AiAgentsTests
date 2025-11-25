import arxiv

# 1. Define the search terms based on the Kaggle competition
# We want the main topic AND the specific method (GO terms)
search_terms = '(abs:"protein function prediction" OR ti:"protein function prediction")'
ontology_terms = '(abs:"gene ontology" OR abs:"GO")'
categories = '(cat:cs.LG OR cat:q-bio.QM OR cat:q-bio.BM)'

# 2. Combine them into one powerful query
final_query = f"{search_terms} AND {ontology_terms} AND {categories}"

print(f"Running query: {final_query}\n")

# 3. Create the search object, sorting by Relevance
# This is the most important part for your request!
search = arxiv.Search(
  query = final_query,
  max_results = 5,  # Let's just get the top 5
  sort_by = arxiv.SortCriterion.Relevance
)

# 4. Loop through the results and print them
print("--- Top 5 Relevant Papers Found ---")
try:
    for result in search.results():
        print(f"\nPaper Title: {result.title}")
        print(f"Published:   {result.published.date()}")
        print(f"PDF Link:    {result.pdf_url}")
        # print(f"Summary:     {result.summary}\n") # Uncomment to see the summary
        print("-" * 20)

except StopIteration:
    print("No results found for this query.")