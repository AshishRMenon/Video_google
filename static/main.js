// ----- custom js ----- //

// hide initial
// $("#searching").hide();
// $("#results-table").hide();
$("#error").hide();

// global
var data = [];

$(function() {
  // sanity check
  console.log( "ready!" );

  // image click
  $(".img").click(function(){
    // removes the previous selected class
    $('.active').removeClass('active');
    // add active class to clicked picture
    $(this).addClass("active");
  });


  // submit click
  $("#submit").click(function() {
    // empty/hide results
    $("#results").empty();
    // $("#results-table").hide();
    $("#error").hide();
    // grab image url
    var image = $(".active").attr("src")
    console.log(image)

    // show searching text
    $("#searching").show();
    console.log("searching...")

    // ajax request
    $.ajax({
      type: "POST",
      url: "/search",
      data : { img : image },
      // handle success
      success: function(result) {
        console.log(result.results);
        var data = result.results

		for (var i = 0; i < data.score.length; i++) {
		  $("#results-table").append('<tr><th><a href="'+data.image[i]+'"><img src="'+data.image[i]+
		    '" class="result-img"></a></th><th>'+data.score[i]+'</th></tr>')
		};
      },
      // handle error
      error: function(error) {
        console.log(error);
        // append to dom
        $("#error").append()
      }
    });



    // show table
$("#results-table").show();

  });

});
