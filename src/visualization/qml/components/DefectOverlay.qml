import QtQuick 2.15

Rectangle {
    property string defectLabel: ""
    property double confidence: 0.0

    color: "transparent"
    border { color: "#ff1744"; width: 2 }

    Rectangle {
        anchors { top: parent.top; right: parent.right; topMargin: -18 }
        width: labelText.implicitWidth + 12
        height: 18
        color: "#ff1744"
        radius: 4

        Text {
            id: labelText
            anchors.centerIn: parent
            text: defectLabel + " " + Math.round(confidence * 100) + "%"
            color: "#ffffff"
            font { pixelSize: 11; bold: true }
        }
    }
}
