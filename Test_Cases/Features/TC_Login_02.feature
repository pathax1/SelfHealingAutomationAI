Feature: ZARA Invalid Login Simulation

  Scenario: Submit invalid credentials on login page
    Given the user has launched the ZARA site and reject cookies
    When the user clicks on the login links
    And the user enters an invalid email and password
    And the user clicks on the login button
    Then an error message should be displayed
