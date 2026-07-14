import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

Item {
    RowLayout {
        anchors { fill: parent; margins: 16 }
        spacing: 16

        // Left: Frame list (280px)
        Rectangle {
            Layout.preferredWidth: 280
            Layout.fillHeight: true
            color: "#16213e"; radius: 12

            ColumnLayout {
                anchors { fill: parent; margins: 8 }
                spacing: 8

                Label {
                    text: "Frame History"
                    font { pixelSize: 16; bold: true }
                    color: "#e4e6eb"
                }

                // Filter toolbar
                RowLayout {
                    Layout.fillWidth: true
                    spacing: 4
                    ComboBox {
                        Layout.fillWidth: true
                        cursorShape: Qt.PointingHandCursor
                        model: ["All", "Last 100", "Last 500", "Last 1000"]
                        currentIndex: 0
                    }
                }

                ListView {
                    id: frameList
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    clip: true
                    model: 100  // Placeholder: bind to history model

                    delegate: Rectangle {
                        width: frameList.width - 4
                        height: 56; radius: 8
                        color: index % 2 === 0 ? "#16213e" : "#1a1a2e"

                        MouseArea {
                            anchors.fill: parent
                            cursorShape: Qt.PointingHandCursor
                            onClicked: frameList.currentIndex = index
                        }

                        RowLayout {
                            anchors { fill: parent; margins: 8 }
                            spacing: 8

                            Rectangle {
                                width: 8; height: 8; radius: 4
                                color: ["#00c853", "#ff1744", "#ff9100"][index % 3]
                            }
                            Label {
                                text: "#042" + (100 - index)
                                color: "#e4e6eb"; font { pixelSize: 13; bold: true }
                            }
                            Item { Layout.fillWidth: true }
                            Label {
                                text: "14:32"
                                color: "#5a6478"; font.pixelSize: 11
                            }
                        }
                    }
                }
            }
        }

        // Right: Frame detail
        Rectangle {
            Layout.fillWidth: true
            Layout.fillHeight: true
            color: "#16213e"; radius: 12

            ColumnLayout {
                anchors { fill: parent; margins: 16 }
                spacing: 12

                // Dual image display
                RowLayout {
                    Layout.fillWidth: true
                    Layout.preferredHeight: 300
                    spacing: 8

                    Rectangle {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        color: "#0f3460"; radius: 8

                        ColumnLayout {
                            anchors.centerIn: parent
                            Label {
                                text: "Raw Image"
                                color: "#a0a8b8"; font.pixelSize: 12
                                anchors.horizontalCenter: parent.horizontalCenter
                            }
                            Rectangle {
                                width: 256; height: 256
                                color: "#1a1a2e"; radius: 4
                            }
                        }
                    }

                    Rectangle {
                        Layout.fillWidth: true
                        Layout.fillHeight: true
                        color: "#0f3460"; radius: 8
                        border { color: "#448aff"; width: 2 }

                        ColumnLayout {
                            anchors.centerIn: parent
                            Label {
                                text: "Surface Image"
                                color: "#a0a8b8"; font.pixelSize: 12
                                anchors.horizontalCenter: parent.horizontalCenter
                            }
                            Rectangle {
                                width: 256; height: 256
                                color: "#1a1a2e"; radius: 4
                            }
                        }
                    }
                }

                Rectangle { Layout.fillWidth: true; height: 1; color: "#1f3460" }

                // Evidence chain
                Label {
                    text: "Evidence Chain"
                    color: "#e4e6eb"; font { pixelSize: 14; bold: true }
                }

                ListView {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    model: inspectionVM ? inspectionVM.evidenceModel : null
                    clip: true

                    delegate: RowLayout {
                        width: parent.width - 8
                        Label {
                            text: model.key + ": " + model.value
                            color: "#a0a8b8"; font.pixelSize: 12
                        }
                    }
                }
            }
        }
    }
}
