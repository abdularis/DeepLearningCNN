var imgLoader = {
    loadCount: 20,
    searchResult: null,
    resultStart: 0,
    resultEnd: 0
}

function populateWithSearchResult() {
    if (imgLoader.resultStart >= imgLoader.resultEnd) return;

    for (var i = imgLoader.resultStart; i < imgLoader.resultEnd; i++) {
        addImageResultItem(imgLoader.searchResult.result[i]);
    }
    imgLoader.resultStart = imgLoader.resultEnd;
    imgLoader.resultEnd = Math.min(imgLoader.resultEnd + imgLoader.loadCount, imgLoader.searchResult.result.length);

    if (imgLoader.resultStart >= imgLoader.resultEnd) {
        var end = '<div id="searchResultEnd">' +
                        '<p>Tidak ada lagi hasil.</p>' +
                    '</div>'
        $('#searchResult').append(end)
        return;
    }
}

function addImageResultItem(image_path) {
    var template = '<div class="image-item">' +
                        '<a href="/?search_by_gallery=' + image_path + '">' +
                            '<img src="static/gallery/'+ image_path +'">' + 
                        '</a>'
                    '</div>';
    $('#searchResult').append(template);
}

$(window).scroll(function() {
    if($(window).scrollTop() == $(document).height() - $(window).height()) {
        populateWithSearchResult();
    }
});