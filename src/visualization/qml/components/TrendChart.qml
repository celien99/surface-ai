import QtQuick 2.15

Rectangle {
    id: chartArea
    color: "#16213e"
    radius: 12

    property var dataPoints: []

    Canvas {
        id: canvas
        anchors { fill: parent; margins: 12 }
        onPaint: {
            var ctx = canvas.getContext("2d")
            ctx.clearRect(0, 0, canvas.width, canvas.height)

            if (dataPoints.length < 2) return

            var maxY = 0, minX = Infinity, maxX = -Infinity
            for (var i = 0; i < dataPoints.length; i++) {
                var pt = dataPoints[i]
                if (pt.y > maxY) maxY = pt.y
                if (pt.x < minX) minX = pt.x
                if (pt.x > maxX) maxX = pt.x
            }
            if (maxY === 0) maxY = 1

            var padX = 30, padY = 20
            var w = canvas.width - padX * 2
            var h = canvas.height - padY * 2
            var xRange = maxX - minX || 1

            // Grid lines
            ctx.strokeStyle = "#1f3460"; ctx.lineWidth = 1
            for (var gi = 0; gi <= 4; gi++) {
                var gy = padY + h * gi / 4
                ctx.beginPath(); ctx.moveTo(padX, gy); ctx.lineTo(canvas.width - padX, gy); ctx.stroke()
            }

            // Line
            ctx.strokeStyle = "#448aff"; ctx.lineWidth = 2
            ctx.beginPath()
            for (var j = 0; j < dataPoints.length; j++) {
                var px = padX + w * (dataPoints[j].x - minX) / xRange
                var py = padY + h * (1 - dataPoints[j].y / maxY)
                if (j === 0) ctx.moveTo(px, py)
                else ctx.lineTo(px, py)
            }
            ctx.stroke()
        }
    }
}
