
# Literature is easy !!!

Import your literature lesson and start chatting with a bot to learn about it!


## Installation

Install with docker command

```bash
  git clone https://github.com/lqkhanh195/LiteratureIsEasy

  docker build -t litiseasy .
```
    
## Running Tests

To run tests on this demo version, you have to create a Cohere API key (https://cohere.com/) and add it to a .env file like this

```bash
    COHERE_API_KEY=API_KEY
```

Then just run 

```bash
    docker run --env-file=.env -p 5001:8051 litiseasy
```

to test it at port 5001 of your localhost.

## How to use

After running all these things, you will need to upload a file(only support txt file for now). Then simply type your question into the chat input box and wait for the answer. You can use some test documents that i provided on the test_data folder to test this project.

## Further improvements:
_ Add more MLOps components.  
_ Solve the limit tokens problem.  
_ Manage uploaded documents so user can upload multiple files, remove file also remove data in vectorstore.  
_ Try more advanced RAG techniques  
_ Add web scraping for more context.  
  
