import React, { Component } from "react"
import FormComponent from "./FormComponent"

class Form extends Component {
    state = {
        firstName: "",
        lastName: "",
        age: "",
        gender: "",
        destination: "",
        isVegan: false,
        isKosher: false,
        isLactoseFree: false
    }

    handleChange = (event) => {
        const { name, value, type, checked } = event.target
        type === "checkbox" ? this.setState({ [name]: checked }) : this.setState({ [name]: value })
    }

    render() {
        return (
            <FormComponent
                handleChange={this.handleChange}
                data={this.state}
            />
        )
    }
}

export default Form