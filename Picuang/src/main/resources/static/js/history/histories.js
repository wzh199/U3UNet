$(function () {
    var i = 0;
    axios.get('/api/year')
        .then(function (yearRes) {
                $.each(yearRes.data, function (key, data) {
                    var year = data;
                    $("#histories").append("" +
                        "<span id='" + year + "'></span>");
                    axios.get('/api/month?year=' + year)
                        .then(function (monthRes) {
                            $.each(monthRes.data, function (key, data) {
                                var month = data;
                                $("#" + year).append("" +
                                    "<span id='" + year + "-" + month + "'></span>");
                                $("#" + year + "-" + month).append("<span id='" + year + "-" + month + "-h" + "'><h2>" + year + " 年 " + month + " 月</h2></span>");
                                $("#" + year + "-" + month).append("<span id='" + year + "-" + month + "-p" + "'></span>");
                                axios.get('/api/day?year=' + year + '&month=' + month)
                                    .then(function (dayRes) {
                                        $.each(dayRes.data, function (key, data) {
                                            var day = data;
                                            $("#" + year + "-" + month + "-p").append("<span id='" + year + "-" + month + "-" + day  + "'><h3>" + day + "日</h3></span>");
                                            var type = ["front", "belowRGB", "belowBinary"];
                                            $.each(type, function (key, t) {

                                                $("#" + year + "-" + month + "-" + day).append("<span id='" + year + "-" + month + "-" + day  + "-" + t +"'><h4>类型" + t + "</h4></span>");

                                                axios.get('/api/list?year=' + year + '&month=' + month + '&day=' + day + '&type=' + t)
                                                    .then(function (listRes) {
                                                        $.each(listRes.data, function (key, data) {
                                                            var list = data;
                                                            $("#" + year + "-" + month + "-" + day + "-" + t).append("<a href='" + data.path + "' target='_blank'><img class='lazyload img-thumbnail' src='background/load.gif' data-src='" + list.path + "' style='width: auto; height: auto; max-width: 128px; max-height: 128px; margin: 8px 4px'></a>");
                                                            if (i == 0) {
                                                                $("#histories").prepend("" +
                                                                    "<h5>历史记录根据您的IP地址（" + data.ip + "）所生成，请及时保存，IP地址更改后历史记录将丢失。</h5>");
                                                            }
                                                            ++i;
                                                            $("#picCount").text(i);
                                                        });

                                                        $(".lazyload").lazyload();
                                                    });
                                            });

                                        });
                                    });
                            });
                        });
                });
        });
});