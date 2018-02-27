$(document).ready(function () {
  // Toggle news category tabs.
  function activaTab(tab){
    $('.category-list a[href="#' + tab + '"]').tab('show');
  };

  // Hit enter in filter box = click submit button.
  $("#filter-inputbox").keypress(function(e){
    if(e.keyCode==13) {
      $("#filter-submit").click();
    }
  });
});

// Apply filter.
function doFilter(){
  var pattern= document.getElementById("filter-inputbox").value;
  var regex = undefined;
  if (pattern.replace(/\s/g, "") == "") { // Show all news items if filter is cleared.
    regex = function(){return true};
  } else { // Compile filter.
    var regex = pat2fn(pattern);
  }
  if (regex == undefined) { // Invalid filter.
    document.getElementById("filter-inputbox").style.backgroundColor = "rgb(255, 0, 0)";
    regex = function(){return false};
  } else {
    document.getElementById("filter-inputbox").style.backgroundColor = "rgb(255, 255, 255)";
  }

  var newsItems = document.getElementsByClassName("newsitem");
  for (var i = 0; i < newsItems.length; i++) {
    var newsItem = newsItems[i];
    if (regex(newsItem.innerHTML)) {
      newsItem.style.display = 'block';
    } else {
      newsItem.style.display = 'none';
    }
  }
};

// Vlad's super filter code, used in MDEX & Scale's dashboards.
var ops    = ['&&', '||', '(', ')'];
var opsset = { };
var opsre  = new RegExp(ops.map(function(op){
        opsset[op] = true;
        return op.replace(/[\-\[\]\/\{\}\(\)\*\+\?\.\\\^\$\|]/g, "\\$&")
}).join("|"), 'g');
var numre  = /^(<|>|<=|>=|=|!=)/;
function pat2fn(pat) {
        if (pat == undefined) return undefined;
        pat = pat.replace(opsre, " $& ");
//      console.log("Preprocess", pat);
        var subpats = pat.split(/\s+/);
        var code = "";
        var needop = false;
        for (var i in subpats) {
//      console.log("Token", subpats[i]);
                if (opsset[subpats[i]]) {
                        code += subpats[i];
                        needop = false;
                } else if (subpats[i] != '') {
                        if (needop) {
                                code += '&&';
                        }
                        if (numre.exec(subpats[i])) {
                                code += "parseFloat(val)" + subpats[i].replace(/^=[^=]/, '=$&');
                        } else {
                                var tst = '!=-1';
                                if (subpats[i].charAt(0) == '!') {
                                        subpats[i] = subpats[i].substring(1);
                                        tst = '==-1';
                                }
                                if (subpats[i] == '') return undefined;
                                code += "val.toString().toLowerCase().indexOf('" + subpats[i].toLowerCase().replace(/([^'\\]*(?:\\.[^'\\]*)*)'/g, "$1\\'") + "')" + tst;
                        }
                        needop = true;
                }
        }
        if (code == '') return undefined;
//      console.log("Compiling", code);
        var fn;
        try {
                fn = new Function("val", "return " + code + ';');
//              console.log("Compiled", fn);
                fn(123);
        } catch (err) {
//              console.log("Threw", err);
                return undefined;
        }
        return fn;
}
