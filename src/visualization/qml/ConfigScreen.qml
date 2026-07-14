import QtQuick 2.15
import QtQuick.Controls 2.15
import QtQuick.Layouts 1.15

Item {
    RowLayout {
        anchors { fill: parent; margins: 16 }
        spacing: 16

        // Left: File selector + YAML tree
        Rectangle {
            Layout.preferredWidth: 240
            Layout.fillHeight: true
            color: "#16213e"; radius: 12

            ColumnLayout {
                anchors { fill: parent; margins: 12 }
                spacing: 8

                Label {
                    text: "Config Files"
                    font { pixelSize: 16; bold: true }
                    color: "#e4e6eb"
                }

                // File selector buttons
                RowLayout {
                    spacing: 4
                    Button {
                        text: "Pipeline"
                        cursorShape: Qt.PointingHandCursor
                        Layout.fillWidth: true
                        flat: true
                        onClicked: configVM.LoadConfig("resources/pipeline.yaml")
                    }
                    Button {
                        text: "Rules"
                        cursorShape: Qt.PointingHandCursor
                        Layout.fillWidth: true
                        flat: true
                    }
                    Button {
                        text: "Tree"
                        cursorShape: Qt.PointingHandCursor
                        Layout.fillWidth: true
                        flat: true
                    }
                }

                Rectangle { Layout.fillWidth: true; height: 1; color: "#1f3460" }

                // YAML structure tree (simplified: show raw as text)
                Label {
                    text: "YAML Structure"
                    color: "#a0a8b8"; font.pixelSize: 12
                }

                ScrollView {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    TextArea {
                        readOnly: true
                        color: "#a0a8b8"
                        font { pixelSize: 11; family: "monospace" }
                        text: configVM ? configVM.pipelineYaml : ""
                        background: Rectangle { color: "transparent" }
                    }
                }
            }
        }

        // Right: YAML editor
        Rectangle {
            Layout.fillWidth: true
            Layout.fillHeight: true
            color: "#0d1117"; radius: 12

            ColumnLayout {
                anchors { fill: parent; margins: 12 }
                spacing: 8

                Label {
                    text: "YAML Editor"
                    font { pixelSize: 14; bold: true }
                    color: "#e4e6eb"
                }

                ScrollView {
                    Layout.fillWidth: true
                    Layout.fillHeight: true
                    TextArea {
                        id: yamlEditor
                        color: "#e4e6eb"
                        font { pixelSize: 13; family: "monospace" }
                        text: configVM ? configVM.pipelineYaml : ""
                        cursorShape: Qt.IBeamCursor
                        background: Rectangle { color: "#0d1117" }
                    }
                }
            }
        }
    }

    // Bottom action bar
    Rectangle {
        anchors { left: parent.left; right: parent.right; bottom: parent.bottom }
        height: 56
        color: "#141e30"

        RowLayout {
            anchors { fill: parent; margins: 12 }
            spacing: 12

            Label {
                text: configVM ? configVM.reloadStatus : "Ready"
                color: "#a0a8b8"; font.pixelSize: 12
            }

            Item { Layout.fillWidth: true }

            Button {
                text: "Validate"
                cursorShape: Qt.PointingHandCursor
                onClicked: configVM.ValidateYaml(yamlEditor.text)
            }

            Button {
                text: "Apply"
                cursorShape: Qt.PointingHandCursor
                onClicked: configVM.ApplyParameterChanges(yamlEditor.text)
            }
        }
    }
}
