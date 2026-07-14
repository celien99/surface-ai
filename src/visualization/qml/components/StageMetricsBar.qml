import QtQuick 2.15

RowLayout {
    property string stageName: ""
    property double latencyMs: 0.0
    property double maxLatencyMs: 100.0
    property double thresholdMs: 50.0

    spacing: 8
    Layout.fillWidth: true

    Label {
        text: stageName
        Layout.preferredWidth: 90
        color: "#a0a8b8"
        font.pixelSize: 13
    }

    Rectangle {
        Layout.fillWidth: true
        height: 6
        radius: 3
        color: "#1f3460"

        Rectangle {
            width: parent.width * Math.min(latencyMs / Math.max(maxLatencyMs, 1), 1.0)
            height: parent.height
            radius: 3
            color: latencyMs > thresholdMs ? "#ff1744" : "#448aff"
            Behavior on width { NumberAnimation { duration: 200; easing.type: Easing.OutCubic } }
            Behavior on color { ColorAnimation { duration: 200 } }
        }
    }

    Label {
        text: latencyMs.toFixed(1) + "ms"
        Layout.preferredWidth: 55
        color: latencyMs > thresholdMs ? "#ff1744" : "#a0a8b8"
        font { pixelSize: 12; family: "monospace" }
    }
}
