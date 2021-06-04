// Store our API endpoint inside queryUrl
var data1 = document.getElementsByName('mydata')[0].content;

if(data1=='dashboard'){
  var queryUrl = 
  `api/v1/failures?id=top`;
}
else {
  var queryUrl = 
  `api/v1/failures`;
}


// Perform a GET request to the query URL
d3.json(queryUrl).then(function(data) {
    
  // Once we get a response, send the data.features object to the createFeatures function
  // createFeatures(data.features);
  
  //console.log(data);

  d3.select("tbody")
  .selectAll("tr")
  .data(data)
  .enter()
  .append("tr")
  .html(function(d) {
    return `<td>${d.primary_element}</td><td>${d.start_time}</td><td>${d.end_time}</td><td>${d.duration}</td><td>${d.difference}</td><td>${d.comments}</td><td>${d.failure}</td>`;
  });

});