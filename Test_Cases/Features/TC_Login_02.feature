Feature: IKEA Invalid Login Simulation

  Scenario: Submit invalid credentials on login page
    Given the user is on the IKEA login page
    When the user enters an invalid email and password
    And clicks on the login button
    Then an error message should be displayed
