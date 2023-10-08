import scrapy


class BookspiderSpider(scrapy.Spider):
    name = "bookspider"
    allowed_domains = ["ketabrah.ir"]
    start_urls = ["https://ketabrah.ir/book-category/%DA%A9%D8%AA%D8%A7%D8%A8-%DA%A9%D9%85%DB%8C%DA%A9-%D8%B1%D9%85%D8%A7%D9%86-%D9%85%D8%B5%D9%88%D8%B1/page-1"]


    def parse(self, response):
        books = response.css('div.book-list div.item')
        
        for book in books:
            url = book.css('a.cover::attr(href)').get()
            book_url = "https://ketabrah.ir/" + url
            
            yield response.follow(book_url, callback=self.parse_book_page)       
        
            
        next_page = response.css('div.paging a.next-page::attr(href)').get()
        
        if next_page is not None:    
            next_page_url = "https://ketabrah.ir/" + next_page
            
            yield response.follow(next_page_url, callback=self.parse)
            
    
    def parse_book_page(self, response):
        name = response.xpath('//*[@id="BookDetails"]/table/tbody/tr[1]/td[2]/text()').get().replace("کتاب", "")
        writer = response.xpath('//*[@id="BookDetails"]/table/tbody/tr[2]/td[2]/a/span/text()').get()
        
        if 'مترجم' == response.xpath('//*[@id="BookDetails"]/table/tbody/tr[3]/td[1]/text()').get():
            translator = response.xpath('//*[@id="BookDetails"]/table/tbody/tr[3]/td[2]/a/span/text()').get()
            publisher = response.xpath('//*[@id="BookDetails"]/table/tbody/tr[4]/td[2]/a/span/text()').get() 
            year_of_publication = response.xpath('//*[@id="BookDetails"]/table/tbody/tr[5]/td[2]/text()').get() 
            book_format = response.xpath('//*[@id="BookDetails"]/table/tbody/tr[6]/td[2]/text()').get() 
            number_of_pages = response.xpath('//*[@id="BookDetails"]/table/tbody/tr[7]/td[2]/text()').get() 
            language = response.xpath('//*[@id="BookDetails"]/table/tbody/tr[8]/td[2]/text()').get() 
            isbn = response.xpath('//*[@id="BookDetails"]/table/tbody/tr[9]/td[2]/span/text()').get() 
            category = response.xpath('//*[@id="BookDetails"]/table/tbody/tr[10]/td[2]/a/text()').get() 
            price = response.xpath('//*[@id="BookDetails"]/div/div[2]/div/span/text()').get()
        else: 
            translator = "-"
            publisher = response.xpath('//*[@id="BookDetails"]/table/tbody/tr[3]/td[2]/a/span/text()').get() 
            year_of_publication = response.xpath('//*[@id="BookDetails"]/table/tbody/tr[4]/td[2]/text()').get() 
            book_format = response.xpath('//*[@id="BookDetails"]/table/tbody/tr[5]/td[2]/text()').get() 
            number_of_pages = response.xpath('//*[@id="BookDetails"]/table/tbody/tr[6]/td[2]/text()').get() 
            language = response.xpath('//*[@id="BookDetails"]/table/tbody/tr[7]/td[2]/text()').get() 
            isbn = response.xpath('//*[@id="BookDetails"]/table/tbody/tr[8]/td[2]/span/text()').get() 
            category = response.xpath('//*[@id="BookDetails"]/table/tbody/tr[9]/td[2]/a/text()').get() 
            price = response.xpath('//*[@id="BookDetails"]/div/div[2]/div/span/text()').get() 
        
        yield {
            'name' : name, 
            'writer' : writer, 
            'translator' : translator, 
            'publisher' : publisher, 
            'year_of_publication' : year_of_publication, 
            'format' : book_format, 
            'number_of_pages' : number_of_pages, 
            'language' : language, 
            'ISBN' : isbn, 
            'category' : category, 
            'price' : price,
        }