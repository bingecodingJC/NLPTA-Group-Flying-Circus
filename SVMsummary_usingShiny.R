
#' This is the file to implement R Shiny package to visualize the result of six classification models, including BoW+SVM(rbf), Tfidf+SVM(rbf), 
#' BoW+LinearSVC, Tfidf+LinearSVC, BoW+Naïve Bayes, and Tfidf+Naïve Bayes.The code will demonstrate different accuracy scores by choosing 
#' different models.
#' 
#' @author: kuochenghao
#' @date: Mar 16 2018

library(shiny)
library(readr)

# read raw data from local csv file
dat1 <- read_csv("~/Google 雲端硬碟/HKU mfin/Nature Language Processing and Text Analytics/Group project/NLP3pre/modelsummary.csv")
day <- dat1$`Return Day`

# ui defines the layout of the page
ui <- fluidPage(
  
  titlePanel("Accuracy Score of Different Model"),
  # sidebarLayout divides the whole page into left part and right part
  sidebarLayout(
    # sidebarPanel defines the left part of the page, which contains three control panels: radioButtons, selectInput, checkboxInput
    sidebarPanel(
      
      radioButtons("radio", label = h3("Select Different Model"),
                   choices = list("BoW+SVM(rbf)","Tfidf+SVM(rbf)", 
                   "BoW+LinearSVC","Tfidf+LinearSVC","BoW+Naïve Bayes",
                   "Tfidf+Naïve Bayes"),selected = "BoW+SVM(rbf)"),
      
      selectInput("vertical","Show vertical line in days of cumulative return:", 
                  choices = unique(dat1$`Return Day`),multiple=TRUE),
      
      checkboxInput("hor", "Show horizontal axis", FALSE)
      
    ),
    # mainPanel defines the right part of the page, which is a lineChart
    mainPanel(
      plotOutput("lineChart")
    )   
  )      
)

# server defines the data operation: there will be an output corresponding to an input
server <- function(input, output) {
  # output: display a plot by using renderPlot
  output$lineChart <- renderPlot({  
    # lineChart data will change according to different input of radioButtons
    chartData <- switch(input$radio,
                        "BoW+SVM(rbf)" = list(dat1$`BoW+SVM(rbf)_in`,dat1$`BoW+SVM(rbf)_out`),
                        "Tfidf+SVM(rbf)" = list(dat1$`Tfidf+SVM(rbf)_in`,dat1$`Tfidf+SVM(rbf)_out`),
                        "BoW+LinearSVC" = list(dat1$`BoW+LinearSVC_in`,dat1$`BoW+LinearSVC_out` ),
                        "Tfidf+LinearSVC"=list(dat1$`Tfidf+LinearSVC_in`,dat1$`Tfidf+LinearSVC_out`),
                        "BoW+Naïve Bayes"=list(dat1$`BoW+Naïve Bayes_in`,dat1$`BoW+Naïve Bayes_out`),
                        "Tfidf+Naïve Bayes"=list(dat1$`Tfidf+Naïve Bayes_in`,dat1$`Tfidf+Naïve Bayes_out`)
    )  
    # lineChart title will change according to different input of radioButtons
    chartTitle <- switch(input$radio,
                         "BoW+SVM(rbf)" = "BoW+SVM(rbf)",
                         "Tfidf+SVM(rbf)"="Tfidf+SVM(rbf)",
                         "BoW+LinearSVC" = "BoW+LinearSVC",
                         "Tfidf+LinearSVC" = "Tfidf+LinearSVC",
                         "BoW+Naïve Bayes"="BoW+Naïve Bayes",
                         "Tfidf+Naïve Bayes"="Tfidf+Naïve Bayes"
    )
    
    yrange <- c(0,1)
    xrange <- range(day)
    
    plot(xrange,yrange,type="n",xlab="# days of cumulative return",ylab="Accuracy Score",cex.lab=1.5,
         main=paste("Accuracy Score shown for", chartTitle),
         sub=c("Note: Best accuracy score from parameter selection"))
    
    # the first line is the insample accuracy score
    lines(day,chartData[[1]],col="aquamarine4",lwd=3,type="b",pch=15,cex=2)
    # the second line is the outsample accuracy score
    lines(day,chartData[[2]],col="firebrick3",lwd=3,type="b",pch=15,cex=2)
    # the text of the first line
    text(day, chartData[[1]],labels = chartData[[1]], cex=1,adj=c(NA,2))
    # the text of the second line
    text(day, chartData[[2]],labels = chartData[[2]], cex=1,adj=c(NA,2))
    
    # abline will depend on SelectInput
    abline(v=input$vertical,lty=2) 
    
    legend("bottomright",c("Outsample Accuracy Score","insample Accuracy Score"), 
           col=c('firebrick3','aquamarine4'),pch=15,ncol=1,bty ="n",cex=1.3)
    
    # abline will depend on CheckboxInput
    if (input$hor) {
      abline(h=0)  
    } 
  },height = 500, width = 600)
  
}

shinyApp(ui = ui, server = server)
