window.addEventListener('load', function() {
  var pTags = document.getElementsByTagName("p");
  var searchText = "Arguments";

  for (var i = 0; i < pTags.length; i++) {
    if (pTags[i].innerHTML === searchText) {
      pTags[i].innerHTML = '';
    }
  }
});

